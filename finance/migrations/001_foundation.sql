-- Review candidate only. Apply explicitly to a NEW isolated Finance test DB.
-- No references, triggers, imports or DDL on public.events or Chief/Reader.
BEGIN;
CREATE SCHEMA finance;
REVOKE ALL ON SCHEMA finance FROM PUBLIC;
CREATE DOMAIN finance.currency AS text CHECK (VALUE ~ '^[A-Z]{3}$');
CREATE DOMAIN finance.money AS numeric CHECK (VALUE NOT IN ('NaN'::numeric,'Infinity'::numeric,'-Infinity'::numeric) AND abs(VALUE)<1e18 AND scale(VALUE)<=12);
CREATE DOMAIN finance.verification AS text CHECK (VALUE IN ('invoice_verified','manual_verified','estimate','unverified'));

CREATE TABLE finance.expense_identity (
 expense_id text PRIMARY KEY,
 source_key text NOT NULL UNIQUE CHECK (source_key<>''),
 created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE finance.expense_revision (
 expense_id text NOT NULL REFERENCES finance.expense_identity,
 revision integer NOT NULL CHECK(revision>0),
 incurred_at timestamptz NOT NULL,
 vendor text NOT NULL CHECK(vendor<>''), description text NOT NULL CHECK(description<>''),
 category text NOT NULL CHECK(category IN ('infrastructure','api_ai','subscriptions','rd_testing','domains','data','tooling','other')),
 original_amount finance.money NOT NULL, currency finance.currency NOT NULL,
 fx_to_gbp finance.money NOT NULL CHECK(fx_to_gbp>0), fx_source text NOT NULL CHECK(fx_source<>''),
 fx_observed_at timestamptz NOT NULL,
 gbp_amount numeric GENERATED ALWAYS AS (original_amount * fx_to_gbp) STORED,
 billing_start timestamptz NOT NULL, billing_end timestamptz NOT NULL CHECK(billing_end>billing_start),
 recurring boolean NOT NULL, service text NOT NULL CHECK(service<>''), environment text NOT NULL CHECK(environment<>''),
 evidence_ref text NOT NULL CHECK(evidence_ref<>''), verification finance.verification NOT NULL,
 verified_by text, correction_reason text,
 recorded_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY(expense_id,revision),
 CHECK(currency<>'GBP' OR fx_to_gbp=1),
 CHECK(verification NOT IN ('invoice_verified','manual_verified') OR coalesce(verified_by,'')<>''),
 CHECK(revision=1 OR coalesce(correction_reason,'')<>'')
);
CREATE FUNCTION finance.expense_sequence() RETURNS trigger LANGUAGE plpgsql SET search_path=pg_catalog,finance AS $$
DECLARE expected integer;
BEGIN
 -- Serialize concurrent corrections on immutable identity, no latest-row race.
 PERFORM 1 FROM finance.expense_identity WHERE expense_id=NEW.expense_id FOR UPDATE;
 SELECT coalesce(max(revision),0)+1 INTO expected FROM finance.expense_revision WHERE expense_id=NEW.expense_id;
 IF NEW.revision<>expected THEN RAISE EXCEPTION 'revision gap or duplicate'; END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER expense_sequence BEFORE INSERT ON finance.expense_revision FOR EACH ROW EXECUTE FUNCTION finance.expense_sequence();
CREATE VIEW finance.effective_expenses AS
 SELECT DISTINCT ON(expense_id) * FROM finance.expense_revision ORDER BY expense_id,revision DESC;
CREATE VIEW finance.verified_expenses AS SELECT * FROM finance.effective_expenses WHERE verification IN ('invoice_verified','manual_verified');
CREATE VIEW finance.lifetime_spend AS
 SELECT coalesce(sum(gbp_amount),0) AS observed_verified_gbp,
 (SELECT count(*) FROM finance.effective_expenses WHERE verification IN ('estimate','unverified')) AS uncertain_expenses
 FROM finance.verified_expenses;
CREATE VIEW finance.spend_by_vendor_category AS SELECT vendor,category,sum(gbp_amount) AS gbp_amount FROM finance.verified_expenses GROUP BY vendor,category;
CREATE VIEW finance.monthly_spend AS SELECT date_trunc('month',incurred_at AT TIME ZONE 'UTC') AS period,category,sum(gbp_amount) AS gbp_amount FROM finance.verified_expenses GROUP BY 1,2;
CREATE VIEW finance.weekly_spend AS SELECT date_trunc('week',incurred_at AT TIME ZONE 'UTC') AS period,category,sum(gbp_amount) AS gbp_amount FROM finance.verified_expenses GROUP BY 1,2;

-- Independent forecast schedules: never inferred from historical invoice totals.
-- Corrections append a schedule superseding the prior row; evidence survives.
-- Save issued report snapshots so later corrections do not rewrite old reports.
CREATE TABLE finance.subscription_schedule (
 schedule_id text PRIMARY KEY, subscription_id text NOT NULL,
 supersedes_schedule_id text UNIQUE REFERENCES finance.subscription_schedule,
 effective_from timestamptz NOT NULL, effective_to timestamptz NOT NULL CHECK(effective_to>effective_from),
 monthly_gbp finance.money NOT NULL CHECK(monthly_gbp>=0),
 verification finance.verification NOT NULL, evidence_ref text NOT NULL CHECK(evidence_ref<>''),
 verified_by text,
 CHECK(verification NOT IN ('invoice_verified','manual_verified') OR coalesce(verified_by,'')<>'')
);
CREATE FUNCTION finance.subscription_correction_check() RETURNS trigger LANGUAGE plpgsql SET search_path=pg_catalog,finance AS $$
BEGIN
 IF NEW.supersedes_schedule_id IS NOT NULL AND NOT EXISTS(SELECT 1 FROM finance.subscription_schedule WHERE schedule_id=NEW.supersedes_schedule_id AND subscription_id=NEW.subscription_id) THEN RAISE EXCEPTION 'subscription correction identity mismatch'; END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER subscription_correction BEFORE INSERT ON finance.subscription_schedule FOR EACH ROW EXECUTE FUNCTION finance.subscription_correction_check();
CREATE VIEW finance.effective_subscriptions AS SELECT s.* FROM finance.subscription_schedule s WHERE NOT EXISTS(SELECT 1 FROM finance.subscription_schedule n WHERE n.supersedes_schedule_id=s.schedule_id);
CREATE FUNCTION finance.run_rate(as_of timestamptz) RETURNS TABLE(month_gbp numeric,year_gbp numeric,day_gbp numeric) LANGUAGE plpgsql STABLE SET search_path=pg_catalog,finance AS $$
DECLARE monthly numeric;
BEGIN
 IF as_of IS NULL THEN RAISE EXCEPTION 'as_of required'; END IF;
 IF EXISTS(SELECT 1 FROM finance.effective_subscriptions WHERE effective_from<=as_of AND as_of<effective_to GROUP BY subscription_id HAVING count(*)>1) THEN
 RAISE EXCEPTION 'overlapping subscription schedules'; END IF;
 SELECT coalesce(sum(monthly_gbp),0) INTO monthly FROM finance.effective_subscriptions WHERE effective_from<=as_of AND as_of<effective_to AND verification IN ('invoice_verified','manual_verified');
 RETURN QUERY SELECT monthly,monthly*12,monthly*12/365.25;
END $$;

CREATE TABLE finance.broker_account (
 account_key text PRIMARY KEY CHECK(account_key<>''), -- opaque internal alias
 broker_environment text NOT NULL CHECK(broker_environment IN ('OANDA_PRACTICE','OANDA_LIVE')),
 system_environment text NOT NULL CHECK(system_environment IN ('DEVELOPMENT','VALIDATION','DEMO','SHADOW','REPLAY','PRODUCTION')),
 financial_reality text NOT NULL CHECK(financial_reality='ACTUAL'), -- actual broker actions, including Practice
 currency finance.currency NOT NULL,
 CHECK(financial_reality='ACTUAL')
);
CREATE TABLE finance.broker_transaction (
 account_key text NOT NULL REFERENCES finance.broker_account,
 transaction_id text NOT NULL CHECK(transaction_id<>''), event_time timestamptz NOT NULL,
 transaction_type text NOT NULL, order_id text, batch_id text, request_id text,
 financial_state text NOT NULL CHECK(financial_state IN ('REALISED','UNREALISED')),
 simulation_type text CHECK(simulation_type IS NULL), -- simulation belongs in a separate future domain
 raw_evidence_ref text NOT NULL CHECK(raw_evidence_ref<>''), raw_sha256 text NOT NULL CHECK(raw_sha256 ~ '^[0-9a-f]{64}$'),
 parser_version text NOT NULL, received_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY(account_key,transaction_id)
);
CREATE TABLE finance.broker_component (
 account_key text NOT NULL, transaction_id text NOT NULL, component_key text NOT NULL CHECK(component_key<>''),
 trade_id text, kind text NOT NULL CHECK(kind IN ('realised_pnl','financing','commission','conversion','other_adjustment','capital_transfer')),
 amount finance.money NOT NULL, -- signed net cash effect, credits +, costs -
 fx_to_gbp finance.money NOT NULL CHECK(fx_to_gbp>0), fx_source text NOT NULL CHECK(fx_source<>''),
 fx_observed_at timestamptz NOT NULL,
 gbp_amount numeric GENERATED ALWAYS AS(amount*fx_to_gbp) STORED,
 PRIMARY KEY(account_key,transaction_id,component_key),
 FOREIGN KEY(account_key,transaction_id) REFERENCES finance.broker_transaction
);
CREATE FUNCTION finance.broker_fx_check() RETURNS trigger LANGUAGE plpgsql SET search_path=pg_catalog,finance AS $$
BEGIN
 IF EXISTS(SELECT 1 FROM finance.broker_account WHERE account_key=NEW.account_key AND currency='GBP') AND NEW.fx_to_gbp<>1 THEN RAISE EXCEPTION 'GBP FX must equal 1'; END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER broker_fx_check BEFORE INSERT ON finance.broker_component FOR EACH ROW EXECUTE FUNCTION finance.broker_fx_check();
CREATE TABLE finance.broker_snapshot (
 snapshot_id text PRIMARY KEY, account_key text NOT NULL REFERENCES finance.broker_account,
 as_of timestamptz NOT NULL, balance finance.money, nav finance.money,
 evidence_ref text NOT NULL, completeness text NOT NULL CHECK(completeness IN ('complete','missing_balance','missing_nav','missing_both')),
 CHECK((completeness='complete' AND balance IS NOT NULL AND nav IS NOT NULL) OR
 (completeness='missing_balance' AND balance IS NULL AND nav IS NOT NULL) OR
 (completeness='missing_nav' AND balance IS NOT NULL AND nav IS NULL) OR
 (completeness='missing_both' AND balance IS NULL AND nav IS NULL))
);
CREATE TABLE finance.broker_trade_action (
 account_key text NOT NULL, transaction_id text NOT NULL, trade_id text NOT NULL CHECK(trade_id<>''),
 action text NOT NULL CHECK(action IN ('OPEN','REDUCE','CLOSE')),
 units finance.money NOT NULL, price finance.money NOT NULL CHECK(price>0),
 reason text NOT NULL, attribution_evidence_ref text NOT NULL,
 PRIMARY KEY(account_key,transaction_id,trade_id,action),
 FOREIGN KEY(account_key,transaction_id) REFERENCES finance.broker_transaction
);
CREATE TABLE finance.engine_event_evidence (
 account_key text NOT NULL REFERENCES finance.broker_account,
 source_event_id text NOT NULL, source_event_type text NOT NULL CHECK(source_event_type IN ('prediction','trade_opened','trade_outcome')),
 signal_id text, broker_trade_id text, event_time timestamptz NOT NULL,
 observed_pnl finance.money, source_evidence_ref text NOT NULL,
 source_sha256 text NOT NULL CHECK(source_sha256 ~ '^[0-9a-f]{64}$'),
 PRIMARY KEY(account_key,source_event_id,source_sha256)
 -- Missing IDs/P&L retained as raw-evidence gaps, reconciler MUST emit exceptions.
);
CREATE TABLE finance.reconciliation_run (
 run_id text PRIMARY KEY, account_key text NOT NULL REFERENCES finance.broker_account,
 period_start timestamptz NOT NULL, period_end timestamptz NOT NULL CHECK(period_end>period_start),
 as_of timestamptz NOT NULL, outcome_due_at timestamptz NOT NULL,
 broker_export_complete boolean NOT NULL, event_export_complete boolean NOT NULL,
 first_transaction_id text, last_transaction_id text, evidence_ref text NOT NULL,
 UNIQUE(run_id,account_key)
);
CREATE TABLE finance.reconciliation_exception (
 run_id text NOT NULL REFERENCES finance.reconciliation_run,
 exception_id text NOT NULL, trade_id text,
 code text NOT NULL CHECK(code IN ('broker_record_missing','opened_missing','prediction_missing','outcome_overdue','pending_eod','pnl_missing','pnl_mismatch','duplicate_identity','missing_identity','outcome_without_broker_closure','component_missing','balance_mismatch','unclassified_transaction','scope_missing','currency_missing','currency_mismatch','broker_state_missing')),
 evidence_ref text NOT NULL, PRIMARY KEY(run_id,exception_id)
);
-- Coverage is an explicit reviewed assertion, not absence of exceptions.
CREATE TABLE finance.coverage_assertion (
 assertion_id text PRIMARY KEY, account_key text NOT NULL REFERENCES finance.broker_account,
 period_start timestamptz NOT NULL, period_end timestamptz NOT NULL CHECK(period_end>period_start),
 expenses_complete boolean NOT NULL, broker_complete boolean NOT NULL,
 reconciliation_run_id text NOT NULL,
 reviewed_by text NOT NULL CHECK(reviewed_by<>''), evidence_ref text NOT NULL CHECK(evidence_ref<>''),
 recorded_at timestamptz NOT NULL DEFAULT now(),
 FOREIGN KEY(reconciliation_run_id,account_key) REFERENCES finance.reconciliation_run(run_id,account_key)
);
CREATE VIEW finance.broker_economics AS
 SELECT a.account_key,a.broker_environment,a.system_environment,a.financial_reality,a.currency,
 t.transaction_id,t.event_time,t.financial_state,t.simulation_type,c.component_key,c.trade_id,c.kind,c.amount,c.gbp_amount
 FROM finance.broker_account a JOIN finance.broker_transaction t USING(account_key) JOIN finance.broker_component c USING(account_key,transaction_id);
CREATE VIEW finance.monthly_broker AS
 SELECT account_key,broker_environment,financial_reality,financial_state,date_trunc('month',event_time AT TIME ZONE 'UTC') AS period,kind,sum(gbp_amount) AS gbp_amount
 FROM finance.broker_economics GROUP BY 1,2,3,4,5,6;
CREATE VIEW finance.weekly_broker AS
 SELECT account_key,broker_environment,financial_reality,financial_state,date_trunc('week',event_time AT TIME ZONE 'UTC') AS period,kind,sum(gbp_amount) AS gbp_amount
 FROM finance.broker_economics GROUP BY 1,2,3,4,5,6;

CREATE FUNCTION finance.period_summary(p_account text,p_start timestamptz,p_end timestamptz)
 RETURNS TABLE(observed_spend_gbp numeric,gross_realised_gbp numeric,financing_gbp numeric,other_costs_gbp numeric,net_trading_gbp numeric,coverage_complete boolean,net_economic_gbp numeric,remaining_break_even_gbp numeric)
 LANGUAGE plpgsql STABLE SET search_path=pg_catalog,finance AS $$
DECLARE spend numeric; gross numeric; financing numeric; costs numeric; complete boolean; reality text; broker_env text; sys_env text; net numeric;
BEGIN
 IF p_start IS NULL OR p_end IS NULL OR p_end<=p_start THEN RAISE EXCEPTION 'valid half-open period required'; END IF;
 SELECT financial_reality,broker_environment,system_environment INTO STRICT reality,broker_env,sys_env FROM finance.broker_account WHERE account_key=p_account;
 SELECT coalesce(sum(gbp_amount),0) INTO spend FROM finance.verified_expenses WHERE incurred_at>=p_start AND incurred_at<p_end;
 SELECT coalesce(sum(gbp_amount) FILTER(WHERE kind='realised_pnl'),0),
 coalesce(sum(gbp_amount) FILTER(WHERE kind='financing'),0),
 coalesce(sum(gbp_amount) FILTER(WHERE kind IN ('commission','conversion','other_adjustment')),0)
 INTO gross,financing,costs FROM finance.broker_economics WHERE account_key=p_account AND financial_state='REALISED' AND event_time>=p_start AND event_time<p_end;
 SELECT EXISTS(SELECT 1 FROM finance.coverage_assertion ca JOIN finance.reconciliation_run rr ON rr.run_id=ca.reconciliation_run_id
 WHERE ca.account_key=p_account AND ca.period_start<=p_start AND ca.period_end>=p_end AND ca.expenses_complete AND ca.broker_complete
 AND rr.broker_export_complete AND rr.event_export_complete AND rr.period_start<=p_start AND rr.period_end>=p_end
 AND ca.assertion_id=(SELECT latest.assertion_id FROM finance.coverage_assertion latest WHERE latest.account_key=p_account AND latest.period_start<=p_start AND latest.period_end>=p_end ORDER BY latest.recorded_at DESC,latest.assertion_id DESC LIMIT 1)
 AND rr.run_id=(SELECT latest.run_id FROM finance.reconciliation_run latest WHERE latest.account_key=p_account AND latest.period_start<=p_start AND latest.period_end>=p_end ORDER BY latest.as_of DESC,latest.run_id DESC LIMIT 1)
 AND NOT EXISTS(SELECT 1 FROM finance.effective_expenses e WHERE e.incurred_at>=p_start AND e.incurred_at<p_end AND e.verification IN ('estimate','unverified'))
 AND NOT EXISTS(SELECT 1 FROM finance.reconciliation_exception x WHERE x.run_id=rr.run_id)) INTO complete;
 net:=gross+financing+costs;
 RETURN QUERY SELECT spend,gross,financing,costs,net,complete,
 CASE WHEN complete AND reality='ACTUAL' AND broker_env='OANDA_LIVE' AND sys_env='PRODUCTION' THEN net-spend END,
 CASE WHEN complete AND reality='ACTUAL' AND broker_env='OANDA_LIVE' AND sys_env='PRODUCTION' THEN greatest(spend-net,0) END;
END $$;

CREATE TABLE finance.report_snapshot (
 report_id text PRIMARY KEY, created_at timestamptz NOT NULL DEFAULT now(),
 formula_version text NOT NULL, source_manifest_id text NOT NULL,
 reviewed_coverage_assertion_id text REFERENCES finance.coverage_assertion,
 bounded_output jsonb NOT NULL CHECK(jsonb_typeof(bounded_output)='object' AND octet_length(bounded_output::text)<=1048576)
);

CREATE FUNCTION finance.reject_mutation() RETURNS trigger LANGUAGE plpgsql SET search_path=pg_catalog AS $$
BEGIN RAISE EXCEPTION 'Finance evidence is append-only; append a correction or new run'; END $$;
DO $$
DECLARE tbl text;
BEGIN
 FOREACH tbl IN ARRAY ARRAY['expense_identity','expense_revision','subscription_schedule','broker_account','broker_transaction','broker_component','broker_snapshot','broker_trade_action','engine_event_evidence','reconciliation_run','reconciliation_exception','coverage_assertion','report_snapshot'] LOOP
 EXECUTE format('CREATE TRIGGER immutable_rows BEFORE UPDATE OR DELETE ON finance.%I FOR EACH ROW EXECUTE FUNCTION finance.reject_mutation()',tbl);
 EXECUTE format('CREATE TRIGGER immutable_truncate BEFORE TRUNCATE ON finance.%I FOR EACH STATEMENT EXECUTE FUNCTION finance.reject_mutation()',tbl);
 END LOOP;
END $$;
REVOKE ALL ON ALL TABLES IN SCHEMA finance FROM PUBLIC;
REVOKE ALL ON ALL FUNCTIONS IN SCHEMA finance FROM PUBLIC;
-- No grants or role creation. Separate reviewed Reader allowlist migration needed.
COMMIT;
