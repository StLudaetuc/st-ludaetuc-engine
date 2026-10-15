# Bounded evidence acquisition — operator-only proposal

DRAFT — NOT APPROVED. This tool was not run against production. The CEO-reported
working internal Render Web Shell is a potential operator route, not a new grant
of database privileges. Keep external PostgreSQL networking unchanged.

## Preconditions

1. A named data owner authorises each instrument/timeframe/environment/receipt
   window, row cap and retention destination. Use existing approved read-only
   identity/PGSERVICE and libpq password/service files on the internal host. No
   credentials appear in a command, report or GitHub. Do not create/change accounts,
   roles, credentials, environment variables or service files under this task.
2. Operator checks current_user/privileges: read-only transaction mode protects
   this tool, but does not turn a privileged role into least privilege. No role or
   grant is provisioned by this proposal. If only an execution-writer identity is
   available, request an approved read identity rather than using it silently.
3. Review exact frozen recovery SHA, install dependencies only in an isolated
   authorised operator environment, not the running engine image. Do not import
   engine.py or start its application; startup performs DDL.
4. Approve a private destination with enough space. The output is still private
   market evidence despite defensive redaction. Never commit raw exports.

## Exact commands

On the operator-approved internal environment, from the checked-out repository
root, with an **already configured** PGSERVICE pointing to an authorised reader:

```
python -m research.historical_recovery.extract \
  --instrument GBPUSD --timeframe 1m --environment production \
  --start 2026-10-13T00:00:00Z --end 2026-10-14T00:00:00Z \
  --batch-size 250 --max-rows 5000 --output /approved/private/gbpusd_1m_receipts_20261013
```

`/approved/private` is an operator-chosen example, not an existing approved path.
The final directory must not exist. Files are created mode 0600, directory 0700.
Maximum acquisition interval seven days, batch <=1000, total <=50000 records,
encoded event <=512 KiB, total encoded source rows <=64 MiB. Byte cap violations
abort without claiming complete evidence. Statement timeout 10s, lock timeout 1s
are connection-local settings inside the read-only transaction, never persisted.
Each SELECT is instrument/timeframe/system-environment constrained and parameterised.
Only market_data is selected; no account, prediction, open/outcome or broker queries.

Window basis is **received_at**, not claimed source time or stored event_time.
This deliberately acquires raw timestamp evidence without using a suspect stored
bar time as truth. A complete receipt snapshot still cannot prove all source bars
or first-arrival times: production retry upserts replace received_at. Source-bar
coverage needs independently reviewed expected slots, capture-start/outage history
and delayed-arrival catch-up evidence. The tool exposes this distinction.

The tool uses one connection/REPEATABLE READ READ ONLY snapshot for count/pages,
verifies transaction mode, paginates event_id using C collation and finally rolls
back. It never commits or applies DDL. Early row cap -> PARTIAL with remaining
cursor; running again with --after-event-id starts a **new snapshot**, never a
continuation claimed complete. Exact replay/comparison requires retained manifests
and matching row digests. Snapshot IDs alone do not authenticate source identity.

## Export contents and privacy

- events.jsonl: copied event envelope and original market payload, with explicit
  redactions; original/redacted event digests are separate.
- manifest.json: scope, source count, pages, cursor, snapshot, identity, timestamps,
  bounded-completeness labels and digests; no DSN/password.
- export_audit.json: actual file SHA-256 digests and capture status. No credentials.

Sensitive key patterns and obvious bearer/URL/connection-value patterns are
redacted recursively; source payload digests preserve evidence of removed values.
This is defence in depth, **not a universal detector of secrets hidden in arbitrary
market fields**. All exports remain private and require human review before any
sharing. Never claim a redacted copy is byte-identical to the original. Full raw
payload remains authoritative in PostgreSQL; when redaction removes a required
source field, trust/eligibility stays unproved. CLI errors are fixed sanitized
messages and do not print psycopg2 exceptions or connection strings.

## Offline recovery projection

Transfer only approved redacted copies into a restricted offline environment:

```
python -m research.historical_recovery.project \
  --export /approved/private/gbpusd_1m_receipts_20261013 \
  --start 2026-10-13T00:00:00Z --end 2026-10-14T00:00:00Z \
  --output /approved/private/gbpusd_1m_source_projection.json
```

Defaults: no approved calendar, no known source versions, no independently bound
event evidence; records remain unproved. No source trust is inferred from a hash.
The optional --reviewed-context JSON must be separately approved, not sourced from
payloads; contract historical_context_v1, environment, source_id, provenance_reference,
known_versions, verified_event_digests, and calendar with version/instrument/timeframe/
start/end/slots/approved/reviewed_empty. Each digest binds the exact redacted bare
event supplied to normalizer, excluding acquisition_* metadata. Reconcile its
source digest with the acquisition manifest through independent operator evidence;
do not simply copy all source digests into a trusted list. A redaction changes the
covered representation and must be explicitly acknowledged.

Projection writes a new private file only, keeps original_event/raw indicators and
does not write to any DB. It verifies all manifest and row hashes before processing.
It deliberately marks source-window completeness false even for complete receipt
snapshots. Controlled supplementary evidence is required to close that gap.

## Evidence requested next

Obtain samples across all 35 streams with original time/time_open/time_close,
original indicators, provider/ticker ID, script/schema versions, confirmation and
original event/receipt/retry metadata. Include start/middle/end of day and both
sides of session breaks and format-change boundaries. Start with one bounded
stream/day; validate export/privacy then approve the remaining 34 stream jobs.
Ask for exact CEO snapshot query, captured-at time, per-stream counts and format
counts, overlap query and bounds. Obtain old/new logger source and active-alert
definitions/version-effective times from the independent source investigation.
No OANDA credentials, trading changes, database networking changes or public
analytics extraction is needed or authorised here.
