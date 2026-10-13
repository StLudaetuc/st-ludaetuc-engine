-- Run AFTER 001_foundation.sql in a fresh, disposable PostgreSQL 18 DB.
-- No production database URL should ever be supplied for this test.
BEGIN;
INSERT INTO finance.expense_identity(expense_id,source_key) VALUES('e1','fixture:1');
INSERT INTO finance.expense_revision(expense_id,revision,incurred_at,vendor,description,category,original_amount,currency,fx_to_gbp,fx_source,fx_observed_at,billing_start,billing_end,recurring,service,environment,evidence_ref,verification,verified_by)
 VALUES('e1',1,'2026-10-01T00:00:00Z','Render','fixture','infrastructure',10,'GBP',1,'identity','2026-10-01T00:00:00Z','2026-10-01T00:00:00Z','2026-11-01T00:00:00Z',true,'engine','DEMO','fixture','invoice_verified','fixture');
INSERT INTO finance.expense_revision(expense_id,revision,incurred_at,vendor,description,category,original_amount,currency,fx_to_gbp,fx_source,fx_observed_at,billing_start,billing_end,recurring,service,environment,evidence_ref,verification,verified_by,correction_reason,recorded_at) SELECT expense_id,2,incurred_at,vendor,description,category,12,currency,fx_to_gbp,fx_source,fx_observed_at,billing_start,billing_end,recurring,service,environment,evidence_ref,verification,verified_by,'correction',recorded_at FROM finance.expense_revision WHERE revision=1;
DO $$ BEGIN
 IF (SELECT observed_verified_gbp FROM finance.lifetime_spend)<>12 THEN RAISE EXCEPTION 'correction double counted'; END IF;
 IF (SELECT count(*) FROM finance.expense_revision)<>2 THEN RAISE EXCEPTION 'lost history'; END IF;
 BEGIN UPDATE finance.expense_revision SET original_amount=1; RAISE EXCEPTION 'update allowed'; EXCEPTION WHEN raise_exception THEN IF SQLERRM='update allowed' THEN RAISE; END IF; END;
 BEGIN DELETE FROM finance.expense_revision; RAISE EXCEPTION 'delete allowed'; EXCEPTION WHEN raise_exception THEN IF SQLERRM='delete allowed' THEN RAISE; END IF; END;
 BEGIN TRUNCATE finance.expense_revision; RAISE EXCEPTION 'truncate allowed'; EXCEPTION WHEN raise_exception THEN IF SQLERRM='truncate allowed' THEN RAISE; END IF; END;
 BEGIN INSERT INTO finance.expense_identity VALUES('e2','fixture:1',now()); RAISE EXCEPTION 'duplicate allowed'; EXCEPTION WHEN unique_violation THEN NULL; END;
END $$;
INSERT INTO finance.broker_account VALUES('fixture-live','OANDA_LIVE','PRODUCTION','ACTUAL','GBP'),('fixture-practice','OANDA_PRACTICE','PRODUCTION','ACTUAL','GBP');
INSERT INTO finance.broker_transaction(account_key,transaction_id,event_time,transaction_type,financial_state,raw_evidence_ref,raw_sha256,parser_version)
 VALUES('fixture-live','1','2026-10-01T00:00:00Z','ORDER_FILL','REALISED','fixture',repeat('a',64),'fixture');
INSERT INTO finance.broker_component(account_key,transaction_id,component_key,kind,amount,fx_to_gbp,fx_source,fx_observed_at)
 VALUES('fixture-live','1','pl','realised_pnl',100,1,'identity','2026-10-01T00:00:00Z'),('fixture-live','1','financing','financing',-3,1,'identity','2026-10-01T00:00:00Z'),('fixture-live','1','commission','commission',-2,1,'identity','2026-10-01T00:00:00Z');
INSERT INTO finance.reconciliation_run VALUES('r','fixture-live','2026-10-01T00:00:00Z','2026-11-01T00:00:00Z','2026-11-02T00:00:00Z','2026-11-02T00:00:00Z',true,true,'1','1','fixture');
INSERT INTO finance.coverage_assertion(assertion_id,account_key,period_start,period_end,expenses_complete,broker_complete,reconciliation_run_id,reviewed_by,evidence_ref) VALUES('c','fixture-live','2026-10-01T00:00:00Z','2026-11-01T00:00:00Z',true,true,'r','fixture','fixture');
DO $$ DECLARE s record; BEGIN
 SELECT * INTO s FROM finance.period_summary('fixture-live','2026-10-01T00:00:00Z','2026-11-01T00:00:00Z');
 IF s.net_trading_gbp<>95 OR s.net_economic_gbp<>83 OR NOT s.coverage_complete OR s.remaining_break_even_gbp<>0 THEN RAISE EXCEPTION 'arithmetic failed'; END IF;
 SELECT * INTO s FROM finance.period_summary('fixture-practice','2026-10-01T00:00:00Z','2026-11-01T00:00:00Z');
 IF s.net_economic_gbp IS NOT NULL THEN RAISE EXCEPTION 'practice contamination'; END IF;
END $$;
INSERT INTO finance.reconciliation_exception VALUES('r','x','1','pnl_missing','fixture');
DO $$ DECLARE s record; BEGIN
 SELECT * INTO s FROM finance.period_summary('fixture-live','2026-10-01T00:00:00Z','2026-11-01T00:00:00Z');
 IF s.coverage_complete OR s.net_economic_gbp IS NOT NULL THEN RAISE EXCEPTION 'incomplete conclusion allowed'; END IF;
END $$;
INSERT INTO finance.subscription_schedule VALUES('s1','Render',NULL,'2026-10-01T00:00:00Z','2026-11-01T00:00:00Z',30,'invoice_verified','fixture','fixture');
DO $$ BEGIN
 IF (SELECT year_gbp FROM finance.run_rate('2026-10-01T00:00:00Z'))<>360 THEN RAISE EXCEPTION 'run rate failed'; END IF;
END $$;
ROLLBACK;
