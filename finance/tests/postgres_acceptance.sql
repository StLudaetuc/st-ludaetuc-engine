-- Local disposable PG18 only, AFTER candidate 001. Never use shared St Ludaetuc DB.
BEGIN;
DO $$ BEGIN
 IF current_setting('server_version_num')::integer < 180000 OR current_setting('server_version_num')::integer >= 190000 OR current_database() NOT LIKE 'finance_%isolated%' THEN
 RAISE EXCEPTION 'fresh isolated Finance PostgreSQL 18 required'; END IF;
END $$;
CREATE FUNCTION pg_temp.assert(ok boolean,message text) RETURNS void LANGUAGE plpgsql AS $$ BEGIN IF ok IS DISTINCT FROM true THEN RAISE EXCEPTION '%',message; END IF; END $$;
SELECT pg_temp.assert(finance.q(0.0000000000005)=0.000000000001,'positive tie');
SELECT pg_temp.assert(finance.q(-0.0000000000005)=-0.000000000001,'negative tie');
SELECT pg_temp.assert(finance.q(123456789.123456789012*0.765432109876)=94497790.577283938656,'FX vector');
SELECT pg_temp.assert(finance.q(360/365.25)=0.985626283368,'run-rate division vector');
SELECT pg_temp.assert((SELECT net_economic_gbp IS NULL FROM finance.project_totals('2026-10-01T00:00:00Z','2026-11-01T00:00:00Z')),'empty UNKNOWN must not be zero');
INSERT INTO finance.expense_identity(expense_id,source_key) VALUES('e1','invoice:fixture');
INSERT INTO finance.expense_revision(expense_id,revision,incurred_at,paid_at,vendor,description,category,original_amount,currency,fx_to_gbp,fx_source,fx_observed_at,billing_start,billing_end,recurring,recognition_policy,service,environment,evidence_ref,verification)
 VALUES('e1',1,'2026-10-01T00:00:00Z','2026-10-01T00:00:00Z','Render','fixture','infrastructure',10,'GBP',1,'identity','2026-10-01T00:00:00Z','2026-10-01T00:00:00Z','2026-11-01T00:00:00Z',true,'straight_line_seconds_v1','engine','DEMO','evidence:fixture','unverified');
INSERT INTO finance.expense_revision(expense_id,revision,incurred_at,paid_at,vendor,description,category,original_amount,currency,fx_to_gbp,fx_source,fx_observed_at,billing_start,billing_end,recurring,recognition_policy,service,environment,evidence_ref,verification,correction_reason)
 SELECT expense_id,2,incurred_at,paid_at,vendor,description,category,12,currency,fx_to_gbp,fx_source,fx_observed_at,billing_start,billing_end,recurring,recognition_policy,service,environment,evidence_ref,verification,'correction' FROM finance.expense_revision WHERE revision=1;
INSERT INTO finance.expense_verification(decision_id,expense_id,revision,decision,reviewed_by,evidence_ref,source_decided_at,server_order)
 VALUES('v1','e1',2,'invoice_verified','fixture','evidence:invoice','2100-01-01Z',999999999);
SELECT pg_temp.assert((SELECT sum(gbp_amount)=12 FROM finance.verified_expenses),'effective verified revision');
INSERT INTO finance.expense_verification(decision_id,expense_id,revision,decision,reviewed_by,evidence_ref,source_decided_at)
 VALUES('v2','e1',2,'withdrawn','fixture','evidence:withdraw','2026-10-01T00:00:00Z');
SELECT pg_temp.assert(NOT EXISTS(SELECT 1 FROM finance.verified_expenses),'later withdrawal beats future source');
SELECT pg_temp.assert((SELECT server_order<999999999 FROM finance.expense_verification WHERE decision_id='v1'),'caller cannot choose server identity');
INSERT INTO finance.expense_verification(decision_id,expense_id,revision,decision,reviewed_by,evidence_ref,source_decided_at)
 VALUES('v3','e1',2,'invoice_verified','fixture','evidence:invoice','2026-10-01T00:00:00Z');
DO $$ BEGIN
 BEGIN UPDATE finance.expense_revision SET original_amount=1; RAISE EXCEPTION 'update allowed'; EXCEPTION WHEN raise_exception THEN IF SQLERRM='update allowed' THEN RAISE; END IF; END;
 BEGIN DELETE FROM finance.expense_revision; RAISE EXCEPTION 'delete allowed'; EXCEPTION WHEN raise_exception THEN IF SQLERRM='delete allowed' THEN RAISE; END IF; END;
 BEGIN TRUNCATE finance.expense_revision; RAISE EXCEPTION 'truncate allowed'; EXCEPTION WHEN raise_exception THEN IF SQLERRM='truncate allowed' THEN RAISE; END IF; END;
 BEGIN INSERT INTO finance.expense_identity(expense_id,source_key) VALUES('e2','invoice:fixture'); RAISE EXCEPTION 'duplicate allowed'; EXCEPTION WHEN unique_violation THEN NULL; END;
END $$;
INSERT INTO finance.broker_account(account_key,broker_environment,system_environment,financial_reality,financial_state,simulation_type,currency)
 VALUES('a','OANDA_LIVE','PRODUCTION','ACTUAL','REALISED',NULL,'GBP'),('b','OANDA_LIVE','PRODUCTION','ACTUAL','REALISED',NULL,'GBP'),('p','OANDA_PRACTICE','PRODUCTION','ACTUAL','REALISED',NULL,'GBP');
INSERT INTO finance.broker_transaction(account_key,transaction_id,event_time,source_received_at,transaction_type,raw_evidence_ref,raw_sha256,parser_version)
 SELECT account_key,'t','2026-10-01T00:00:00Z','2026-10-01T00:00:00Z','ORDER_FILL','evidence:tx',repeat('a',64),'fixture' FROM finance.broker_account;
INSERT INTO finance.broker_component(account_key,transaction_id,component_key,kind,amount,fx_to_gbp,fx_source,fx_observed_at)
 SELECT account_key,'t','pl','realised_pnl',CASE account_key WHEN 'a' THEN 100 WHEN 'b' THEN 50 ELSE 9999 END,1,'identity','2026-10-01T00:00:00Z' FROM finance.broker_account;
INSERT INTO finance.broker_component(account_key,transaction_id,component_key,kind,amount,fx_to_gbp,fx_source,fx_observed_at)
 SELECT account_key,'t',c.key,c.kind,c.amount,1,'identity','2026-10-01T00:00:00Z' FROM finance.broker_account CROSS JOIN (VALUES('financing','financing',-3),('commission','commission',-2),('guaranteedExecutionFee','other_adjustment',0)) c(key,kind,amount);
INSERT INTO finance.engine_event_evidence(account_key,source_event_id,source_event_type,event_time,source_received_at,fill_status,source_evidence_ref,source_sha256)
 SELECT account_key,'ev','trade_opened','2026-10-01T00:00:00Z','2026-10-01T00:00:00Z','FILLED','evidence:ev',repeat('b',64) FROM finance.broker_account;
INSERT INTO finance.reconciliation_run(run_id,account_key,period_start,period_end,source_as_of,outcome_due_at,broker_evidence_count,internal_evidence_count,broker_complete,internal_complete,broker_reviewed_empty,internal_reviewed_empty,evidence_ref)
 SELECT 'r-'||account_key,account_key,'2026-10-01T00:00:00Z','2026-11-01T00:00:00Z','2100-01-01Z','2026-11-02T00:00:00Z',1,1,true,true,false,false,'evidence:run' FROM finance.broker_account;
INSERT INTO finance.reconciliation_result(run_id,result_id,trade_id,state,reason,evidence_ref)
 SELECT 'r-'||account_key,'matched','trade','MATCHED','matched all evidence','evidence:matched' FROM finance.broker_account;
INSERT INTO finance.coverage_assertion(assertion_id,domain,scope_key,period_start,period_end,state,evidence_count,first_evidence_id,last_evidence_id,reconciliation_run_id,currency_confirmed,costs_complete,reviewed_by,evidence_ref,source_asserted_at)
 SELECT 'c-'||account_key,'BROKER',account_key,'2026-10-01T00:00:00Z','2026-11-01T00:00:00Z','VERIFIED',1,'t','t','r-'||account_key,true,true,'fixture','manifest:broker','2100-01-01Z' FROM finance.broker_account;
INSERT INTO finance.coverage_assertion(assertion_id,domain,scope_key,period_start,period_end,state,evidence_count,first_evidence_id,last_evidence_id,currency_confirmed,costs_complete,reviewed_by,evidence_ref)
 VALUES('expenses','EXPENSES','PROJECT','2026-10-01T00:00:00Z','2026-11-01T00:00:00Z','VERIFIED',1,'e1','e1',true,true,'fixture','manifest:expenses'),
 ('inventory','PROJECT_INVENTORY','PROJECT','2026-10-01T00:00:00Z','2026-11-01T00:00:00Z','VERIFIED',3,'a','p',true,true,'fixture','manifest:inventory');
SELECT pg_temp.assert((SELECT net_economic_gbp=128 AND cash_net_economic_gbp=128 FROM finance.project_totals('2026-10-01T00:00:00Z','2026-11-01T00:00:00Z')),'95+45-12 once, Practice excluded');
INSERT INTO finance.coverage_assertion(assertion_id,domain,scope_key,period_start,period_end,state,evidence_count,currency_confirmed,costs_complete,reviewed_by,evidence_ref,source_asserted_at)
 VALUES('withdraw','BROKER','a','2026-10-01T00:00:00Z','2026-11-01T00:00:00Z','WITHDRAWN',1,true,true,'fixture','manifest:withdraw','2026-10-01T00:00:00Z');
SELECT pg_temp.assert((SELECT net_economic_gbp IS NULL FROM finance.project_totals('2026-10-01T00:00:00Z','2026-11-01T00:00:00Z')),'withdrawal supersedes future source');
INSERT INTO finance.reconciliation_run(run_id,account_key,period_start,period_end,source_as_of,outcome_due_at,broker_evidence_count,internal_evidence_count,broker_complete,internal_complete,broker_reviewed_empty,internal_reviewed_empty,evidence_ref)
 VALUES('new-a','a','2026-10-01T00:00:00Z','2026-11-01T00:00:00Z','2026-10-02T00:00:00Z','2026-11-02T00:00:00Z',1,1,true,true,false,false,'evidence:new');
INSERT INTO finance.reconciliation_result(run_id,result_id,trade_id,state,reason,evidence_ref)
 VALUES('new-a','bad','trade','VALUE_MISMATCH','broker discrepancy','evidence:bad');
SELECT pg_temp.assert((SELECT run_id='new-a' FROM finance.latest_runs WHERE account_key='a'),'latest run not source timestamp');
INSERT INTO finance.subscription_schedule(schedule_id,subscription_id,effective_from,effective_to,monthly_gbp,verification,evidence_ref)
 VALUES('s','Render','2026-10-01T00:00:00Z','2026-11-01T00:00:00Z',30,'invoice_verified','evidence:subscription');
INSERT INTO finance.coverage_assertion(assertion_id,domain,scope_key,period_start,period_end,state,evidence_count,first_evidence_id,last_evidence_id,currency_confirmed,costs_complete,reviewed_by,evidence_ref)
 VALUES('recurring','RECURRING','PROJECT','2026-10-01T00:00:00Z','2026-11-01T00:00:00Z','VERIFIED',1,'s','s',true,true,'fixture','manifest:recurring');
SELECT pg_temp.assert((SELECT year_gbp=360 AND state='VERIFIED' FROM finance.run_rate('2026-10-01T00:00:00Z','2026-10-01T00:00:00Z','2026-11-01T00:00:00Z')),'run rate policy');
SELECT pg_temp.assert(finance.q(365 * extract(epoch FROM '2026-11-01T00:00:00Z'::timestamptz-'2026-10-01T00:00:00Z'::timestamptz)/extract(epoch FROM '2027-10-01T00:00:00Z'::timestamptz-'2026-10-01T00:00:00Z'::timestamptz))=31,'prepaid service recognition vector');
ROLLBACK;
