-- UNDEPLOYED candidate replacement; fresh isolated PG18 only. No startup DDL.
BEGIN;
CREATE SCHEMA finance;
REVOKE ALL ON SCHEMA finance FROM PUBLIC;
CREATE SEQUENCE finance.authority_order;
CREATE DOMAIN finance.money AS numeric CHECK(VALUE NOT IN ('NaN'::numeric,'Infinity'::numeric,'-Infinity'::numeric) AND abs(VALUE)<1e18 AND scale(VALUE)<=12);
CREATE DOMAIN finance.currency AS text CHECK(VALUE IN ('GBP','USD','EUR','AED','CAD','JPY','AUD','CHF'));
CREATE FUNCTION finance.q(value numeric) RETURNS numeric LANGUAGE sql IMMUTABLE STRICT SET search_path=pg_catalog AS $$ SELECT round(value,12) $$;
CREATE FUNCTION finance.q_ratio(n numeric,d numeric) RETURNS numeric LANGUAGE plpgsql IMMUTABLE STRICT SET search_path=pg_catalog AS $$
DECLARE k integer;nn numeric;dd numeric;whole numeric;rem numeric;
BEGIN
 IF d=0 THEN RAISE EXCEPTION 'zero denominator'; END IF;
 k:=greatest(scale(n),scale(d)); nn:=abs(n)*power(10::numeric,k+12); dd:=abs(d)*power(10::numeric,k);
 whole:=div(nn,dd);rem:=mod(nn,dd);
 IF rem*2>=dd THEN whole:=whole+1; END IF;
 RETURN sign(n)*sign(d)*whole*0.000000000001;
END $$;
-- PostgreSQL numeric round ties away from zero; Python ROUND_HALF_UP equivalent.
CREATE TABLE finance.expense_identity(expense_id text PRIMARY KEY, source_key text NOT NULL UNIQUE, server_order bigint NOT NULL);
CREATE TABLE finance.expense_revision(
 expense_id text NOT NULL REFERENCES finance.expense_identity, revision integer NOT NULL CHECK(revision>0),
 incurred_at timestamptz NOT NULL, paid_at timestamptz, vendor text NOT NULL, description text NOT NULL,
 category text NOT NULL CHECK(category IN ('infrastructure','api_ai','subscriptions','rd_testing','domains','data','tooling','other')),
 original_amount finance.money NOT NULL, currency finance.currency NOT NULL,
 fx_to_gbp finance.money NOT NULL CHECK(fx_to_gbp>0), fx_source text NOT NULL, fx_observed_at timestamptz NOT NULL,
 gbp_amount numeric GENERATED ALWAYS AS(finance.q(original_amount*fx_to_gbp)) STORED,
 billing_start timestamptz NOT NULL, billing_end timestamptz NOT NULL CHECK(billing_end>billing_start),
 recurring boolean NOT NULL, recognition_policy text CHECK(recognition_policy IN ('straight_line_seconds_v1','point_in_time_v1')),
 service text NOT NULL, environment text NOT NULL, evidence_ref text NOT NULL,
 verification text NOT NULL CHECK(verification IN ('unverified','estimate')), -- importer cannot self-verify
 correction_reason text, source_recorded_at timestamptz,
 server_order bigint NOT NULL, server_received_at timestamptz NOT NULL,
 PRIMARY KEY(expense_id,revision), CHECK(currency<>'GBP' OR fx_to_gbp=1),
 CHECK(evidence_ref ~ '^(invoice|receipt|evidence|manifest|attestation|statement|source|billing|fx|assumption):[A-Za-z0-9_.-]{1,160}$'),
 CHECK(revision=1 OR coalesce(correction_reason,'')<>'')
);
CREATE TABLE finance.expense_verification(
 decision_id text PRIMARY KEY, expense_id text NOT NULL, revision integer NOT NULL,
 decision text NOT NULL CHECK(decision IN ('invoice_verified','manual_verified','unverified','estimate','withdrawn')),
 reviewed_by text NOT NULL CHECK(reviewed_by<>''), evidence_ref text NOT NULL CHECK(evidence_ref ~ '^(invoice|receipt|evidence|manifest|attestation|statement|source|billing|fx|assumption):[A-Za-z0-9_.-]{1,160}$'),
 source_decided_at timestamptz, server_order bigint NOT NULL, server_received_at timestamptz NOT NULL,
 FOREIGN KEY(expense_id,revision) REFERENCES finance.expense_revision
);
CREATE FUNCTION finance.expense_sequence() RETURNS trigger LANGUAGE plpgsql SET search_path=pg_catalog,finance AS $$
DECLARE expected integer;
BEGIN
 PERFORM 1 FROM finance.expense_identity WHERE expense_id=NEW.expense_id FOR UPDATE;
 SELECT coalesce(max(revision),0)+1 INTO expected FROM finance.expense_revision WHERE expense_id=NEW.expense_id;
 IF NEW.revision<>expected THEN RAISE EXCEPTION 'revision gap or duplicate'; END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER expense_sequence BEFORE INSERT ON finance.expense_revision FOR EACH ROW EXECUTE FUNCTION finance.expense_sequence();
CREATE VIEW finance.effective_expenses WITH (security_invoker=true) AS SELECT DISTINCT ON(expense_id) * FROM finance.expense_revision ORDER BY expense_id,revision DESC;
CREATE VIEW finance.latest_verifications WITH (security_invoker=true) AS SELECT DISTINCT ON(expense_id) * FROM finance.expense_verification ORDER BY expense_id,server_order DESC;
CREATE VIEW finance.verified_expenses WITH (security_invoker=true) AS SELECT e.* FROM finance.effective_expenses e JOIN finance.latest_verifications v USING(expense_id)
 WHERE v.revision=e.revision AND v.decision IN ('invoice_verified','manual_verified');

CREATE TABLE finance.subscription_schedule(
 schedule_id text PRIMARY KEY, subscription_id text NOT NULL, supersedes_schedule_id text UNIQUE REFERENCES finance.subscription_schedule,
 effective_from timestamptz NOT NULL,effective_to timestamptz NOT NULL CHECK(effective_to>effective_from),
 monthly_gbp finance.money NOT NULL CHECK(monthly_gbp>=0),
 verification text NOT NULL CHECK(verification IN ('invoice_verified','manual_verified','estimate','unverified')),
 evidence_ref text NOT NULL, source_at timestamptz, server_order bigint NOT NULL, server_received_at timestamptz NOT NULL
);
CREATE FUNCTION finance.subscription_identity() RETURNS trigger LANGUAGE plpgsql SET search_path=pg_catalog,finance AS $$
BEGIN
 IF NEW.supersedes_schedule_id IS NOT NULL AND NOT EXISTS(SELECT 1 FROM finance.subscription_schedule WHERE schedule_id=NEW.supersedes_schedule_id AND subscription_id=NEW.subscription_id) THEN RAISE EXCEPTION 'subscription identity mismatch'; END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER subscription_identity BEFORE INSERT ON finance.subscription_schedule FOR EACH ROW EXECUTE FUNCTION finance.subscription_identity();
CREATE VIEW finance.effective_subscriptions WITH (security_invoker=true) AS SELECT s.* FROM finance.subscription_schedule s WHERE NOT EXISTS(SELECT 1 FROM finance.subscription_schedule n WHERE n.supersedes_schedule_id=s.schedule_id);

CREATE TABLE finance.broker_account(
 account_key text PRIMARY KEY,
 broker_environment text NOT NULL CHECK(broker_environment IN ('OANDA_PRACTICE','OANDA_LIVE')),
 system_environment text NOT NULL CHECK(system_environment IN ('DEVELOPMENT','VALIDATION','DEMO','SHADOW','REPLAY','PRODUCTION')),
 financial_reality text NOT NULL CHECK(financial_reality IN ('ACTUAL','SIMULATED')),
 financial_state text NOT NULL CHECK(financial_state IN ('REALISED','UNREALISED')),
 simulation_type text,
 currency finance.currency NOT NULL, server_order bigint NOT NULL,
 CHECK((financial_reality='ACTUAL' AND simulation_type IS NULL) OR
 (financial_reality='SIMULATED' AND financial_state='UNREALISED' AND simulation_type IS NOT NULL AND simulation_type IN ('COUNTERFACTUAL','SHADOW','REPLAY','VALIDATION')))
);
CREATE TABLE finance.broker_transaction(
 account_key text NOT NULL REFERENCES finance.broker_account,transaction_id text NOT NULL,
 event_time timestamptz NOT NULL, source_received_at timestamptz NOT NULL, transaction_type text NOT NULL,
 order_id text,batch_id text,request_id text,
 raw_evidence_ref text NOT NULL,raw_sha256 text NOT NULL CHECK(raw_sha256 ~ '^[0-9a-f]{64}$'),parser_version text NOT NULL,
 server_order bigint NOT NULL,server_received_at timestamptz NOT NULL,
 PRIMARY KEY(account_key,transaction_id)
);
CREATE TABLE finance.broker_component(
 account_key text NOT NULL,transaction_id text NOT NULL,component_key text NOT NULL,trade_id text,
 kind text NOT NULL CHECK(kind IN ('realised_pnl','financing','commission','conversion','other_adjustment','capital_transfer')),
 amount finance.money NOT NULL,fx_to_gbp finance.money NOT NULL CHECK(fx_to_gbp>0),fx_source text NOT NULL,fx_observed_at timestamptz NOT NULL,
 gbp_amount numeric GENERATED ALWAYS AS(finance.q(amount*fx_to_gbp)) STORED,
 server_order bigint NOT NULL,
 PRIMARY KEY(account_key,transaction_id,component_key),FOREIGN KEY(account_key,transaction_id) REFERENCES finance.broker_transaction
);
CREATE FUNCTION finance.broker_fx() RETURNS trigger LANGUAGE plpgsql SET search_path=pg_catalog,finance AS $$
BEGIN
 IF EXISTS(SELECT 1 FROM finance.broker_account WHERE account_key=NEW.account_key AND currency='GBP') AND NEW.fx_to_gbp<>1 THEN RAISE EXCEPTION 'GBP FX must equal 1'; END IF;
 IF NOT EXISTS(SELECT 1 FROM finance.broker_transaction t WHERE t.account_key=NEW.account_key AND t.transaction_id=NEW.transaction_id AND
 ((t.transaction_type='ORDER_FILL' AND ((NEW.component_key='pl' AND NEW.kind='realised_pnl') OR (NEW.component_key='financing' AND NEW.kind='financing') OR
 (NEW.component_key='commission' AND NEW.kind='commission' AND NEW.amount<=0) OR (NEW.component_key='guaranteedExecutionFee' AND NEW.kind='other_adjustment' AND NEW.amount<=0))) OR
 (t.transaction_type='DAILY_FINANCING' AND NEW.component_key='financing' AND NEW.kind='financing') OR
 (t.transaction_type='TRANSFER_FUNDS' AND NEW.component_key='amount' AND NEW.kind='capital_transfer'))) THEN RAISE EXCEPTION 'unreviewed cash component/profile'; END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER broker_fx BEFORE INSERT ON finance.broker_component FOR EACH ROW EXECUTE FUNCTION finance.broker_fx();
CREATE TABLE finance.broker_trade_action(
 account_key text NOT NULL,transaction_id text NOT NULL,trade_id text NOT NULL,
 action text NOT NULL CHECK(action IN ('OPEN','REDUCE','CLOSE')), units finance.money NOT NULL,price finance.money NOT NULL CHECK(price>0),
 attribution_pl finance.money,reason text NOT NULL,evidence_ref text NOT NULL,server_order bigint NOT NULL,
 PRIMARY KEY(account_key,transaction_id,trade_id,action),FOREIGN KEY(account_key,transaction_id) REFERENCES finance.broker_transaction
);
CREATE TABLE finance.broker_snapshot(
 snapshot_id text PRIMARY KEY,account_key text NOT NULL REFERENCES finance.broker_account,as_of timestamptz NOT NULL,
 balance finance.money,nav finance.money,evidence_ref text NOT NULL,
 completeness text NOT NULL CHECK(completeness IN ('complete','missing_balance','missing_nav','missing_both')),server_order bigint NOT NULL,
 CHECK((completeness='complete' AND balance IS NOT NULL AND nav IS NOT NULL) OR
 (completeness='missing_balance' AND balance IS NULL AND nav IS NOT NULL) OR
 (completeness='missing_nav' AND balance IS NOT NULL AND nav IS NULL) OR
 (completeness='missing_both' AND balance IS NULL AND nav IS NULL))
);
CREATE TABLE finance.engine_event_evidence(
 account_key text NOT NULL REFERENCES finance.broker_account,source_event_id text NOT NULL,
 source_event_type text NOT NULL CHECK(source_event_type IN ('prediction','trade_opened','trade_outcome')),
 signal_id text,broker_trade_id text,event_time timestamptz NOT NULL,source_received_at timestamptz NOT NULL,
 fill_status text CHECK(fill_status IN ('FILLED','CANCELLED','REJECTED','UNKNOWN')),observed_pnl finance.money,
 source_evidence_ref text NOT NULL,source_sha256 text NOT NULL CHECK(source_sha256 ~ '^[0-9a-f]{64}$'),server_order bigint NOT NULL,
 PRIMARY KEY(account_key,source_event_id)
);
CREATE TABLE finance.reconciliation_run(
 run_id text PRIMARY KEY,account_key text NOT NULL REFERENCES finance.broker_account,
 period_start timestamptz NOT NULL,period_end timestamptz NOT NULL CHECK(period_end>period_start),
 source_as_of timestamptz NOT NULL,outcome_due_at timestamptz NOT NULL,
 schedule_version text,schedule_evidence_ref text,schedule_verified boolean NOT NULL DEFAULT false,
 CHECK(NOT schedule_verified OR (schedule_version='london_eod_tues_sat_0030_v1' AND schedule_evidence_ref ~ '^evidence:[A-Za-z0-9_.-]{1,160}$')),
 broker_evidence_count integer NOT NULL CHECK(broker_evidence_count>=0),internal_evidence_count integer NOT NULL CHECK(internal_evidence_count>=0),
 broker_complete boolean NOT NULL,internal_complete boolean NOT NULL,
 broker_reviewed_empty boolean NOT NULL,internal_reviewed_empty boolean NOT NULL,
 CHECK((broker_evidence_count=0 AND (NOT broker_complete OR broker_reviewed_empty)) OR (broker_evidence_count>0 AND NOT broker_reviewed_empty)),
 CHECK((internal_evidence_count=0 AND (NOT internal_complete OR internal_reviewed_empty)) OR (internal_evidence_count>0 AND NOT internal_reviewed_empty)),
 evidence_ref text NOT NULL,server_order bigint NOT NULL,server_received_at timestamptz NOT NULL,UNIQUE(run_id,account_key)
);
CREATE TABLE finance.reconciliation_result(
 run_id text NOT NULL REFERENCES finance.reconciliation_run,result_id text NOT NULL,trade_id text,
 state text NOT NULL CHECK(state IN ('MATCHED','BROKER_ONLY','INTERNAL_ONLY','VALUE_MISMATCH','PENDING_EOD_SYNC','INVESTIGATION_REQUIRED')),
 reason text NOT NULL,evidence_ref text NOT NULL,server_order bigint NOT NULL,PRIMARY KEY(run_id,result_id)
);
CREATE TABLE finance.reconciliation_binding(
 run_id text NOT NULL,result_id text NOT NULL,account_key text NOT NULL,
 transaction_id text,source_event_id text,
 FOREIGN KEY(run_id,result_id) REFERENCES finance.reconciliation_result,
 FOREIGN KEY(run_id,account_key) REFERENCES finance.reconciliation_run(run_id,account_key),
 FOREIGN KEY(account_key,transaction_id) REFERENCES finance.broker_transaction,
 FOREIGN KEY(account_key,source_event_id) REFERENCES finance.engine_event_evidence(account_key,source_event_id),
 CHECK((transaction_id IS NULL)<>(source_event_id IS NULL)),
 UNIQUE(run_id,transaction_id),UNIQUE(run_id,source_event_id),server_order bigint NOT NULL
);
CREATE FUNCTION finance.binding_evidence() RETURNS trigger LANGUAGE plpgsql SET search_path=pg_catalog,finance AS $$
DECLARE r record;trade text;
BEGIN
 SELECT * INTO r FROM finance.reconciliation_run WHERE run_id=NEW.run_id AND account_key=NEW.account_key;
 SELECT trade_id INTO trade FROM finance.reconciliation_result WHERE run_id=NEW.run_id AND result_id=NEW.result_id;
 IF NEW.transaction_id IS NOT NULL AND NOT EXISTS(SELECT 1 FROM finance.broker_transaction tx WHERE tx.account_key=NEW.account_key AND tx.transaction_id=NEW.transaction_id AND tx.event_time>=r.period_start AND tx.event_time<r.period_end) THEN RAISE EXCEPTION 'broker binding outside scope/window'; END IF;
 IF NEW.source_event_id IS NOT NULL AND NOT EXISTS(SELECT 1 FROM finance.engine_event_evidence ev WHERE ev.account_key=NEW.account_key AND ev.source_event_id=NEW.source_event_id AND ev.event_time>=r.period_start AND ev.event_time<r.period_end AND (ev.broker_trade_id IS NULL OR ev.broker_trade_id=trade)) THEN RAISE EXCEPTION 'internal binding outside trade scope/window'; END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER binding_evidence BEFORE INSERT ON finance.reconciliation_binding FOR EACH ROW EXECUTE FUNCTION finance.binding_evidence();
CREATE TABLE finance.exception_resolution(
 decision_id text PRIMARY KEY,run_id text NOT NULL,result_id text NOT NULL,
 state text NOT NULL CHECK(state IN ('RESOLVED','WITHDRAWN')),matched_run_id text,matched_result_id text,
 reviewed_by text NOT NULL CHECK(reviewed_by<>''),evidence_ref text NOT NULL CHECK(evidence_ref<>''),source_decided_at timestamptz,
 server_order bigint NOT NULL,server_received_at timestamptz NOT NULL,
 FOREIGN KEY(run_id,result_id) REFERENCES finance.reconciliation_result,
 FOREIGN KEY(matched_run_id,matched_result_id) REFERENCES finance.reconciliation_result,
 CHECK((state='RESOLVED' AND matched_run_id IS NOT NULL AND matched_result_id IS NOT NULL) OR (state='WITHDRAWN' AND matched_run_id IS NULL AND matched_result_id IS NULL))
);
CREATE FUNCTION finance.resolution_evidence() RETURNS trigger LANGUAGE plpgsql SET search_path=pg_catalog,finance AS $$
BEGIN
 IF NOT EXISTS(SELECT 1 FROM finance.reconciliation_result WHERE run_id=NEW.run_id AND result_id=NEW.result_id AND state<>'MATCHED') THEN RAISE EXCEPTION 'source exception required'; END IF;
 IF NEW.state='RESOLVED' AND NOT EXISTS(SELECT 1 FROM finance.reconciliation_result n JOIN finance.reconciliation_run nr ON nr.run_id=n.run_id
 JOIN finance.reconciliation_result src ON src.run_id=NEW.run_id AND src.result_id=NEW.result_id
 JOIN finance.reconciliation_run oldr ON oldr.run_id=src.run_id
 WHERE n.run_id=NEW.matched_run_id AND n.result_id=NEW.matched_result_id AND n.state='MATCHED'
 AND n.trade_id=src.trade_id AND nr.account_key=oldr.account_key AND nr.server_order>oldr.server_order AND nr.period_start=oldr.period_start AND nr.period_end=oldr.period_end
 AND NOT EXISTS(SELECT 1 FROM finance.reconciliation_run newer WHERE newer.account_key=nr.account_key AND newer.period_start=nr.period_start AND newer.period_end=nr.period_end AND newer.server_order>nr.server_order)) THEN RAISE EXCEPTION 'new matched same-account evidence required'; END IF;
 IF NEW.state='WITHDRAWN' AND NOT EXISTS(SELECT 1 FROM finance.exception_resolution d WHERE d.run_id=NEW.run_id AND d.result_id=NEW.result_id AND d.state='RESOLVED' AND d.server_order=(SELECT max(server_order) FROM finance.exception_resolution WHERE run_id=NEW.run_id AND result_id=NEW.result_id)) THEN RAISE EXCEPTION 'current resolved decision required for withdrawal'; END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER resolution_evidence BEFORE INSERT ON finance.exception_resolution FOR EACH ROW EXECUTE FUNCTION finance.resolution_evidence();
CREATE VIEW finance.latest_resolutions WITH (security_invoker=true) AS SELECT DISTINCT ON(run_id,result_id) * FROM finance.exception_resolution ORDER BY run_id,result_id,server_order DESC;
CREATE VIEW finance.unresolved_results WITH (security_invoker=true) AS SELECT r.* FROM finance.reconciliation_result r LEFT JOIN finance.latest_resolutions d USING(run_id,result_id)
 WHERE r.state<>'MATCHED' AND coalesce(d.state,'WITHDRAWN')<>'RESOLVED';

CREATE TABLE finance.coverage_assertion(
 assertion_id text PRIMARY KEY,domain text NOT NULL CHECK(domain IN ('BROKER','EXPENSES','RECURRING','PROJECT_INVENTORY')),
 scope_key text NOT NULL,period_start timestamptz NOT NULL,period_end timestamptz NOT NULL CHECK(period_end>period_start),
 state text NOT NULL CHECK(state IN ('VERIFIED','REVIEWED_EMPTY','UNKNOWN','INCOMPLETE','WITHDRAWN')),
 evidence_count integer NOT NULL CHECK(evidence_count>=0),first_evidence_id text,last_evidence_id text,
 reconciliation_run_id text REFERENCES finance.reconciliation_run,
 currency_confirmed boolean NOT NULL,costs_complete boolean NOT NULL,
 reviewed_by text NOT NULL,evidence_ref text NOT NULL,source_asserted_at timestamptz,
 server_order bigint NOT NULL,server_received_at timestamptz NOT NULL,
 CHECK(state<>'VERIFIED' OR (evidence_count>0 AND first_evidence_id IS NOT NULL AND last_evidence_id IS NOT NULL)),
 CHECK(state<>'REVIEWED_EMPTY' OR (evidence_count=0 AND first_evidence_id IS NULL AND last_evidence_id IS NULL)),
 CHECK(state NOT IN ('VERIFIED','REVIEWED_EMPTY') OR reviewed_by<>''),
 CHECK(evidence_ref ~ '^(invoice|receipt|evidence|manifest|attestation|statement|source|billing|fx|assumption):[A-Za-z0-9_.-]{1,160}$')
);
CREATE VIEW finance.latest_coverage WITH (security_invoker=true) AS SELECT DISTINCT ON(domain,scope_key,period_start,period_end) * FROM finance.coverage_assertion ORDER BY domain,scope_key,period_start,period_end,server_order DESC;
CREATE VIEW finance.latest_runs WITH (security_invoker=true) AS SELECT DISTINCT ON(account_key,period_start,period_end) * FROM finance.reconciliation_run ORDER BY account_key,period_start,period_end,server_order DESC;
CREATE FUNCTION finance.coverage_ok(p_domain text,p_scope text,p_start timestamptz,p_end timestamptz,p_count bigint)
 RETURNS boolean LANGUAGE sql STABLE SET search_path=pg_catalog,finance AS $$
 SELECT coalesce((SELECT c.state IN ('VERIFIED','REVIEWED_EMPTY') AND c.evidence_count=p_count AND c.currency_confirmed AND c.costs_complete AND c.period_start<=p_start AND c.period_end>=p_end
 AND (p_domain<>'BROKER' OR EXISTS(SELECT 1 FROM finance.latest_runs r WHERE r.run_id=c.reconciliation_run_id AND r.account_key=p_scope
 AND r.schedule_verified AND r.broker_complete AND r.internal_complete AND r.broker_evidence_count=p_count
 AND r.internal_evidence_count=(SELECT count(*) FROM finance.engine_event_evidence ev WHERE ev.account_key=p_scope AND ev.event_time>=r.period_start AND ev.event_time<r.period_end)
 AND NOT EXISTS(SELECT 1 FROM finance.broker_transaction tx WHERE tx.account_key=p_scope AND tx.event_time>=p_start AND tx.event_time<p_end AND NOT EXISTS(SELECT 1 FROM finance.reconciliation_binding b WHERE b.run_id=r.run_id AND b.account_key=p_scope AND b.transaction_id=tx.transaction_id))
 AND NOT EXISTS(SELECT 1 FROM finance.engine_event_evidence ev WHERE ev.account_key=p_scope AND ev.event_time>=p_start AND ev.event_time<p_end AND NOT EXISTS(SELECT 1 FROM finance.reconciliation_binding b WHERE b.run_id=r.run_id AND b.account_key=p_scope AND b.source_event_id=ev.source_event_id))
 AND NOT EXISTS(SELECT 1 FROM finance.broker_trade_action ba JOIN finance.broker_transaction tx USING(account_key,transaction_id) WHERE ba.account_key=p_scope AND tx.event_time>=p_start AND tx.event_time<p_end AND NOT EXISTS(SELECT 1 FROM finance.reconciliation_result rr WHERE rr.run_id=r.run_id AND rr.trade_id=ba.trade_id))
 AND NOT EXISTS(SELECT 1 FROM finance.unresolved_results prior JOIN finance.reconciliation_run pr ON pr.run_id=prior.run_id WHERE pr.account_key=p_scope AND pr.period_start=p_start AND pr.period_end=p_end AND pr.server_order<r.server_order) AND r.period_start<=p_start AND r.period_end>=p_end
 AND NOT EXISTS(SELECT 1 FROM finance.unresolved_results x WHERE x.run_id=r.run_id)))
 FROM finance.latest_coverage c WHERE c.domain=p_domain AND c.scope_key=p_scope AND c.period_start=p_start AND c.period_end=p_end),false)
 $$;
CREATE VIEW finance.broker_economics WITH (security_invoker=true) AS SELECT a.account_key,a.broker_environment,a.system_environment,a.financial_reality,a.financial_state,a.simulation_type,a.currency,
 t.transaction_id,t.event_time,c.component_key,c.trade_id,c.kind,c.amount,c.gbp_amount
 FROM finance.broker_account a JOIN finance.broker_transaction t USING(account_key) JOIN finance.broker_component c USING(account_key,transaction_id);
CREATE FUNCTION finance.account_totals(p_account text,p_start timestamptz,p_end timestamptz)
 RETURNS TABLE(observed_net_gbp numeric,net_trading_gbp numeric,coverage_complete boolean,evidence_count bigint)
 LANGUAGE plpgsql STABLE SET search_path=pg_catalog,finance AS $$
DECLARE n bigint;net numeric;ok boolean;
BEGIN
 IF p_start IS NULL OR p_end IS NULL OR p_end<=p_start OR NOT EXISTS(SELECT 1 FROM finance.broker_account WHERE account_key=p_account) THEN RAISE EXCEPTION 'explicit valid scope/window required'; END IF;
 SELECT count(*) INTO n FROM finance.broker_transaction WHERE account_key=p_account AND event_time>=p_start AND event_time<p_end;
 SELECT finance.q(coalesce(sum(gbp_amount),0)) INTO net FROM finance.broker_economics WHERE account_key=p_account AND event_time>=p_start AND event_time<p_end AND kind<>'capital_transfer';
 ok:=finance.coverage_ok('BROKER',p_account,p_start,p_end,n);
 -- No unknown monetary transaction or missing top-level cash profile may pass.
 IF EXISTS(SELECT 1 FROM finance.broker_transaction t WHERE t.account_key=p_account AND t.event_time>=p_start AND t.event_time<p_end AND
 (t.transaction_type NOT IN ('ORDER_FILL','DAILY_FINANCING','TRANSFER_FUNDS','MARKET_ORDER','ORDER_CANCEL','MARKET_ORDER_REJECT','LIMIT_ORDER','LIMIT_ORDER_REJECT','STOP_ORDER','STOP_ORDER_REJECT','TAKE_PROFIT_ORDER','TAKE_PROFIT_ORDER_REJECT','STOP_LOSS_ORDER','STOP_LOSS_ORDER_REJECT','TRAILING_STOP_LOSS_ORDER','TRAILING_STOP_LOSS_ORDER_REJECT') OR
 (SELECT count(*) FROM finance.broker_component c WHERE c.account_key=t.account_key AND c.transaction_id=t.transaction_id AND
 c.component_key=ANY(CASE t.transaction_type WHEN 'ORDER_FILL' THEN ARRAY['pl','financing','commission','guaranteedExecutionFee'] WHEN 'DAILY_FINANCING' THEN ARRAY['financing'] ELSE ARRAY['amount'] END))
 <> CASE t.transaction_type WHEN 'ORDER_FILL' THEN 4 WHEN 'DAILY_FINANCING' THEN 1 WHEN 'TRANSFER_FUNDS' THEN 1 ELSE 0 END)) THEN ok:=false; END IF;
 RETURN QUERY SELECT net,CASE WHEN ok THEN net END,ok,n;
END $$;
CREATE FUNCTION finance.expense_totals(p_start timestamptz,p_end timestamptz)
 RETURNS TABLE(cash_spend_gbp numeric,economic_expense_gbp numeric,coverage_complete boolean,evidence_count bigint,policy_missing bigint)
 LANGUAGE plpgsql STABLE SET search_path=pg_catalog,finance AS $$
DECLARE n bigint;v bigint;unknown_policy bigint;unknown_cash bigint;ok boolean;cash numeric;expense numeric;
BEGIN
 IF p_start IS NULL OR p_end IS NULL OR p_end<=p_start THEN RAISE EXCEPTION 'valid half-open period required'; END IF;
 SELECT count(*) INTO n FROM finance.effective_expenses;
 SELECT count(*),count(*) FILTER(WHERE recognition_policy IS NULL),count(*) FILTER(WHERE paid_at IS NULL),
 finance.q(coalesce(sum(gbp_amount) FILTER(WHERE paid_at>=p_start AND paid_at<p_end),0)),
 finance.q(coalesce(sum(CASE recognition_policy WHEN 'point_in_time_v1' THEN CASE WHEN incurred_at>=p_start AND incurred_at<p_end THEN gbp_amount ELSE 0 END
 WHEN 'straight_line_seconds_v1' THEN finance.q_ratio(gbp_amount * least(extract(epoch FROM billing_end-billing_start),greatest(extract(epoch FROM p_end-billing_start),0)),extract(epoch FROM billing_end-billing_start))
 - finance.q_ratio(gbp_amount * least(extract(epoch FROM billing_end-billing_start),greatest(extract(epoch FROM p_start-billing_start),0)),extract(epoch FROM billing_end-billing_start)) END),0))
 INTO v,unknown_policy,unknown_cash,cash,expense FROM finance.verified_expenses;
 ok:=finance.coverage_ok('EXPENSES','PROJECT',p_start,p_end,n) AND n=v;
 RETURN QUERY SELECT CASE WHEN ok AND unknown_cash=0 THEN cash END,CASE WHEN ok AND unknown_policy=0 THEN expense END,ok,n,unknown_policy;
END $$;
CREATE FUNCTION finance.project_totals(p_start timestamptz,p_end timestamptz)
 RETURNS TABLE(actual_net_trading_gbp numeric,cash_spend_gbp numeric,economic_expense_gbp numeric,cash_net_economic_gbp numeric,net_economic_gbp numeric,remaining_cash_break_even_gbp numeric,coverage_complete boolean)
 LANGUAGE plpgsql STABLE SET search_path=pg_catalog,finance AS $$
DECLARE a record;e record;t record;n bigint;net numeric:=0;ok boolean;
BEGIN
 SELECT count(*) INTO n FROM finance.broker_account;
 ok:=finance.coverage_ok('PROJECT_INVENTORY','PROJECT',p_start,p_end,n);
 FOR a IN SELECT account_key FROM finance.broker_account WHERE broker_environment='OANDA_LIVE' AND system_environment='PRODUCTION' AND financial_reality='ACTUAL' AND financial_state='REALISED' AND simulation_type IS NULL LOOP
 SELECT * INTO t FROM finance.account_totals(a.account_key,p_start,p_end);
 ok:=ok AND t.coverage_complete;net:=net+coalesce(t.net_trading_gbp,0);
 END LOOP;
 SELECT * INTO e FROM finance.expense_totals(p_start,p_end);
 RETURN QUERY SELECT CASE WHEN ok THEN finance.q(net) END,e.cash_spend_gbp,e.economic_expense_gbp,
 CASE WHEN ok AND e.coverage_complete AND e.cash_spend_gbp IS NOT NULL THEN finance.q(net-e.cash_spend_gbp) END,
 CASE WHEN ok AND e.coverage_complete AND e.economic_expense_gbp IS NOT NULL THEN finance.q(net-e.economic_expense_gbp) END,
 CASE WHEN ok AND e.coverage_complete AND e.cash_spend_gbp IS NOT NULL THEN finance.q(greatest(e.cash_spend_gbp-net,0)) END,
 ok AND e.coverage_complete AND e.cash_spend_gbp IS NOT NULL;
END $$;
CREATE FUNCTION finance.run_rate(p_as_of timestamptz,p_start timestamptz,p_end timestamptz)
 RETURNS TABLE(month_gbp numeric,year_gbp numeric,day_gbp numeric,state text,excluded_estimates bigint,unknown_records bigint)
 LANGUAGE plpgsql STABLE SET search_path=pg_catalog,finance AS $$
DECLARE n bigint;excluded bigint;unknown_count bigint;monthly numeric;ok boolean;
BEGIN
 IF p_as_of IS NULL OR p_start IS NULL OR p_end IS NULL OR p_end<=p_start OR p_as_of<p_start OR p_as_of>=p_end THEN RAISE EXCEPTION 'as_of and reviewed period required'; END IF;
 IF EXISTS(SELECT 1 FROM finance.effective_subscriptions WHERE effective_from<=p_as_of AND effective_to>p_as_of GROUP BY subscription_id HAVING count(*)>1) THEN RAISE EXCEPTION 'overlapping subscription schedules'; END IF;
 SELECT count(*),count(*) FILTER(WHERE verification='estimate'),count(*) FILTER(WHERE verification='unverified'),
 finance.q(coalesce(sum(monthly_gbp) FILTER(WHERE verification IN ('invoice_verified','manual_verified')),0)) INTO n,excluded,unknown_count,monthly
 FROM finance.effective_subscriptions WHERE effective_from<=p_as_of AND effective_to>p_as_of;
 ok:=finance.coverage_ok('RECURRING','PROJECT',p_start,p_end,n) AND excluded=0 AND unknown_count=0;
 RETURN QUERY SELECT CASE WHEN ok THEN monthly END,CASE WHEN ok THEN finance.q(monthly*12) END,CASE WHEN ok THEN finance.q_ratio(monthly*12,365.25) END,
 CASE WHEN ok THEN 'VERIFIED' ELSE 'UNKNOWN' END,excluded,unknown_count;
END $$;
CREATE FUNCTION finance.cash_spend_periods(p_grain text,p_timezone text)
 RETURNS TABLE(period timestamp,category text,observed_gbp numeric) LANGUAGE plpgsql STABLE SET search_path=pg_catalog,finance AS $$
BEGIN
 IF p_grain NOT IN ('week','month') OR p_timezone IS NULL OR NOT EXISTS(SELECT 1 FROM pg_timezone_names WHERE name=p_timezone) THEN RAISE EXCEPTION 'explicit reporting calendar required'; END IF;
 RETURN QUERY SELECT date_trunc(p_grain,e.paid_at AT TIME ZONE p_timezone),e.category,finance.q(sum(e.gbp_amount)) FROM finance.verified_expenses e WHERE e.paid_at IS NOT NULL GROUP BY 1,2;
END $$;
CREATE VIEW finance.order_denominators WITH (security_invoker=true) AS SELECT account_key,count(*) AS attempted_orders,count(*) FILTER(WHERE fill_status='FILLED') AS filled_orders,count(*) FILTER(WHERE fill_status='CANCELLED') AS cancelled_orders
 FROM finance.engine_event_evidence WHERE source_event_type='trade_opened' GROUP BY account_key;
CREATE TABLE finance.report_snapshot(report_id text PRIMARY KEY,formula_version text NOT NULL,source_manifest_id text NOT NULL,
 bounded_output jsonb NOT NULL CHECK(jsonb_typeof(bounded_output)='object' AND octet_length(bounded_output::text)<=1048576),server_order bigint NOT NULL,server_received_at timestamptz NOT NULL);
CREATE TABLE finance.authority_allocation(server_order bigint PRIMARY KEY);
CREATE FUNCTION finance.lock_authority() RETURNS trigger LANGUAGE plpgsql SET search_path=pg_catalog,finance AS $$
DECLARE key text;head bigint;expected bigint;
BEGIN
 IF current_setting('transaction_isolation')<>'read committed' THEN RAISE EXCEPTION 'authority decisions require READ COMMITTED'; END IF;
 CASE TG_TABLE_NAME
 WHEN 'expense_verification' THEN key:=NEW.expense_id;
 WHEN 'coverage_assertion' THEN key:=NEW.domain||':'||NEW.scope_key||':'||extract(epoch FROM NEW.period_start)::text||':'||extract(epoch FROM NEW.period_end)::text;
 WHEN 'reconciliation_run' THEN key:=NEW.account_key||':'||extract(epoch FROM NEW.period_start)::text||':'||extract(epoch FROM NEW.period_end)::text;
 WHEN 'exception_resolution' THEN key:=NEW.run_id||':'||NEW.result_id;
 END CASE;
 PERFORM pg_advisory_xact_lock(hashtextextended(TG_TABLE_NAME||':'||key,0));
 IF TG_TABLE_NAME='expense_verification' THEN SELECT max(server_order) INTO head FROM finance.expense_verification WHERE expense_id=NEW.expense_id;
 ELSIF TG_TABLE_NAME='coverage_assertion' THEN SELECT max(server_order) INTO head FROM finance.coverage_assertion WHERE domain=NEW.domain AND scope_key=NEW.scope_key AND period_start=NEW.period_start AND period_end=NEW.period_end;
 ELSIF TG_TABLE_NAME='reconciliation_run' THEN SELECT max(server_order) INTO head FROM finance.reconciliation_run WHERE account_key=NEW.account_key AND period_start=NEW.period_start AND period_end=NEW.period_end;
 ELSE SELECT max(server_order) INTO head FROM finance.exception_resolution WHERE run_id=NEW.run_id AND result_id=NEW.result_id;
 END IF;
 expected:=NEW.supersedes_server_order;
 IF expected IS NOT NULL AND expected<>coalesce(head,0) THEN RAISE EXCEPTION 'stale authority predecessor'; END IF;
 NEW.supersedes_server_order:=head;
 RETURN NEW;
END $$;
DO $$ DECLARE t text; BEGIN
 FOREACH t IN ARRAY ARRAY['expense_verification','coverage_assertion','reconciliation_run','exception_resolution'] LOOP
 EXECUTE format('ALTER TABLE finance.%I ADD COLUMN supersedes_server_order bigint',t);
 EXECUTE format('CREATE TRIGGER "0_authority_lock" BEFORE INSERT ON finance.%I FOR EACH ROW EXECUTE FUNCTION finance.lock_authority()',t);
 END LOOP;
END $$;
CREATE FUNCTION finance.stamp_authority() RETURNS trigger LANGUAGE plpgsql SET search_path=pg_catalog,finance AS $$
BEGIN
 -- Forced even if a caller explicitly supplies server_order/server_received_at.
 NEW.server_order:=nextval('finance.authority_order');
 INSERT INTO finance.authority_allocation VALUES(NEW.server_order);
 IF TG_ARGV[0]='clock' THEN NEW.server_received_at:=clock_timestamp(); END IF;
 RETURN NEW;
END $$;
CREATE FUNCTION finance.reject_mutation() RETURNS trigger LANGUAGE plpgsql SET search_path=pg_catalog AS $$
BEGIN RAISE EXCEPTION 'append-only evidence; append a decision, revision or new run'; END $$;
DO $$
DECLARE t text;clock text;
BEGIN
 FOREACH t IN ARRAY ARRAY['expense_identity','expense_revision','expense_verification','subscription_schedule','broker_account','broker_transaction','broker_component','broker_trade_action','broker_snapshot','engine_event_evidence','reconciliation_run','reconciliation_result','reconciliation_binding','exception_resolution','coverage_assertion','report_snapshot'] LOOP
 EXECUTE format('ALTER TABLE finance.%I ADD UNIQUE(server_order)',t);
 clock:=CASE WHEN t IN ('expense_revision','expense_verification','subscription_schedule','broker_transaction','reconciliation_run','exception_resolution','coverage_assertion','report_snapshot') THEN 'clock' ELSE 'order' END;
 EXECUTE format('CREATE TRIGGER a_server_order BEFORE INSERT ON finance.%I FOR EACH ROW EXECUTE FUNCTION finance.stamp_authority(%L)',t,clock);
 EXECUTE format('CREATE TRIGGER immutable_rows BEFORE UPDATE OR DELETE ON finance.%I FOR EACH ROW EXECUTE FUNCTION finance.reject_mutation()',t);
 EXECUTE format('CREATE TRIGGER immutable_truncate BEFORE TRUNCATE ON finance.%I FOR EACH STATEMENT EXECUTE FUNCTION finance.reject_mutation()',t);
 END LOOP;
END $$;
CREATE TRIGGER immutable_rows BEFORE UPDATE OR DELETE ON finance.authority_allocation FOR EACH ROW EXECUTE FUNCTION finance.reject_mutation();
CREATE TRIGGER immutable_truncate BEFORE TRUNCATE ON finance.authority_allocation FOR EACH STATEMENT EXECUTE FUNCTION finance.reject_mutation();
REVOKE ALL ON ALL TABLES IN SCHEMA finance FROM PUBLIC;
REVOKE ALL ON ALL SEQUENCES IN SCHEMA finance FROM PUBLIC;
REVOKE ALL ON ALL FUNCTIONS IN SCHEMA finance FROM PUBLIC;
-- No service/Chief grants. Separate trusted verification writer and staging importer required.
COMMIT;
