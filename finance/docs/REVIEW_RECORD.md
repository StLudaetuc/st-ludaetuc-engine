# Finance correction review record

Old reviewed candidate: `189587ce3e211b22782dc221af9afdf6a8cc123c`.
Independent verdict on old candidate:
`FINANCE_FOUNDATION_INDEPENDENT_REVIEW_REQUIRES_CORRECTION`.

Only `StLudaetuc/st-ludaetuc-engine`, `finance/foundation-v1` changed. Existing
Draft PR #2 remains the sole PR. Main, other branches, execution logic, Chief/
recovery-controller, Render, shared PostgreSQL, Make, TradingView and OANDA are
unchanged. No merge/deployment/cloud validation DB.

The commit containing this record is the corrected source candidate; the delivery
handover records its exact SHA after Git assigns it. Independent review of that
SHA remains pending; this provider does not approve its own work.

Validation: 152 local checks passed, including the 67 baseline case names,
regressions and PG18 grammar checks. Original 67 cases passed before any edit.
Scope/evidence-dependent fixtures and changed error expectations were updated to
the corrected APIs; no original case was deleted. Runtime PG18 acceptance,
SQL/Python parity and DB concurrency/privilege validation are NOT_YET_DEMONSTRATED.

See HANDOVER.md for correction evidence, explicit accounting-policy choices and
named residuals. Review readiness is source/design review only, not activation.
