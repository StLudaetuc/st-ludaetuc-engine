# Canonical encoding and independent expected evidence — v1.1 DRAFT

## Canonical timestamp / digest boundary

canonical_json_value_v2_utc6 is a **new incompatible hash version**. Old v1 hashes
must never be relabelled/reused as v2. Typed JSON encoding remains deterministic,
with aware Python datetime values converted to UTC `YYYY-MM-DDTHH:MM:SS.ffffffZ`.
Exactly six fractional digits, no float epoch conversion or receipt-time fallback.
Datetime JSON export uses that same format, so Decimal + timestamp round trips
preserve value digests regardless of the PostgreSQL session timezone. Aware source
payload ISO strings remain original strings in raw payloads; their lexical evidence
is never silently rewritten. Timestamptz stores an instant, not original source
submission offset; preserving raw payloads is essential to retain that offset.

ISO source timestamps exceeding six fractional digits are explicitly rejected,
not truncated by Python. Millisecond source fields reject float, exponent and
fractional lexical forms, even if the value is mathematically integral. Integer
and digit-string forms still need explicit allowed date range and correct units.
No real historical timestamp has been verified by these changes.

## Independent expectation versus internal consistency

The export producer writes manifest, per-record/source/payload hashes and file audit.
Those can all be recomputed by a malicious producer. Verifying them establishes
only internal consistency, not source authenticity. project_export now verifies
actual events/manifest file hashes against export_audit.json and all internal
row/population hashes. It then checks a separate TrustedAnchor when supplied.

An anchor pins: expected manifest hash, expected audit-file SHA-256, source/exported
counts, source scope/window, acquired-at, operator reference, manifest and redaction
versions, and ordered original IDs + original/redacted event/payload digests. The
anchor custodian independently retains these expectations through an approved
channel outside the producer's export directory. The implementation **does not**
issue/approve/sign anchors. CLI rejects anchor files placed inside the export tree.
No production key or trust service is provisioned by this task.

All anchor expectations must match. Wrong manifest/audit/record/count/scope/window/
identity/version returns a fixed failure code before evidence promotion. No anchor
returns UNPROVED, clears any supplied per-row provenance grants in the projection
and leaves all real research eligibility blocked. --require-anchor makes absent
anchors a hard failure. A Context/Calendar file alone cannot grant exported data
verified provenance without a matching anchor. Receipt snapshots still do not
become complete source-bar windows even after a successful match.

`MATCHED_INDEPENDENT_EXPECTATION` describes comparison against verifier-supplied
expectations, not cryptographic proof of who supplied the file. Producer-authored
anchors falsely presented as independently held remain a custody/governance attack;
secure distribution/retention and custodian verification must be approved before
real acquisition. This is not an alternative ledger or replacement for PostgreSQL.
Do not serve these offline APIs directly to untrusted callers.

Future CLI pattern:

```
python -m research.historical_recovery.project --export /private/export \
  --start 2026-10-13T00:00:00Z --end 2026-10-14T00:00:00Z \
  --trusted-anchor /independent-custodian/expected.json --require-anchor \
  --reviewed-context /approved-registry/context.json --output /private/projection.json
```

Paths are examples, not authorised destinations or permission to extract. Test-only
make_anchor helpers are conspicuously synthetic and must never be a real trust issuer.
