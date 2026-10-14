-- DRAFT — NOT APPROVED. Review-only least-privilege design; never startup DDL.
-- Requires pre-created NOLOGIN roles finance_stager / finance_verifier /
-- finance_reconciler on a local disposable test DB. No roles created here.
-- No permissions are installed on Render/shared systems by this candidate.
GRANT USAGE ON SCHEMA finance TO finance_stager,finance_verifier,finance_reconciler;
GRANT USAGE ON SEQUENCE finance.authority_order TO finance_stager,finance_verifier,finance_reconciler;
GRANT INSERT ON finance.authority_allocation TO finance_stager,finance_verifier,finance_reconciler;
GRANT SELECT ON finance.expense_identity,finance.expense_revision TO finance_stager;
-- SELECT FOR UPDATE in the revision trigger requires UPDATE privilege. Direct
-- UPDATE still hits immutable_rows and is rejected; role must not own objects.
GRANT UPDATE ON finance.expense_identity TO finance_stager;
GRANT INSERT ON finance.expense_identity,finance.expense_revision TO finance_stager;
GRANT SELECT ON finance.expense_revision,finance.expense_verification TO finance_verifier;
GRANT INSERT ON finance.expense_verification TO finance_verifier;
GRANT SELECT ON finance.broker_account,finance.broker_transaction,finance.broker_component,
 finance.broker_trade_action,finance.engine_event_evidence,finance.reconciliation_run,
 finance.reconciliation_result,finance.reconciliation_binding,finance.exception_resolution
 TO finance_reconciler;
GRANT INSERT ON finance.reconciliation_run,finance.reconciliation_result,
 finance.reconciliation_binding,finance.exception_resolution TO finance_reconciler;
-- PostgreSQL checks trigger-function EXECUTE when a trigger is installed, but
-- direct deterministic helper calls need explicit EXECUTE; no SECURITY DEFINER.
GRANT EXECUTE ON FUNCTION finance.q(numeric) TO finance_stager;
-- No coverage publisher, broker ingester, subscription verifier or Reader role
-- is commissioned. Such roles need separately approved scoped identity/RLS.
-- Do not grant ownership, CREATE, TRUNCATE, setval, or BYPASSRLS to service roles.
