# Future isolated SELECT-only acquisition — v1.1 DRAFT, NOT AUTHORISED TO RUN

**Do not use the existing Render engine Web Shell.** The independent reviewer
reports it contains production writer credentials. This task neither accessed
that shell nor provisioned a replacement identity/environment. No live extraction
is authorised. PostgreSQL external networking must remain unchanged.

## Future design subject to CEO approval

Use a separate isolated internal execution environment and a dedicated login with
SELECT on the approved market-data source only, no table writes/DDL/elevated role
flags or writable membership, and no trading credential inheritance. A data owner
must review effective grants, SET ROLE memberships, SECURITY DEFINER/function
execution, other object privileges and network egress. Read-only transaction mode
alone does not turn an engine writer identity into least privilege. Provisioning
and independent privilege validation are separate future work, not performed here.

After approval, operators install reviewed code/dependencies in the isolated host,
use an already provisioned libpq PGSERVICE/password-service mechanism and a private
destination. No DSN/credential is accepted on this tool's command line or printed.

Proposed command, not permission to execute:

```
python -m research.historical_recovery.extract \
  --instrument GBPUSD --timeframe 1m --environment production \
  --start 2026-10-13T00:00:00Z --end 2026-10-14T00:00:00Z \
  --batch-size 250 --max-rows 5000 \
  --approved-reader-login dedicated_market_reader \
  --approved-execution-context CEO_APPROVED_ISOLATED_JOB_REFERENCE \
  --output /approved/private/gbpusd_1m_export
```

These names/paths are placeholders, not created resources or approved identities.
The CLI requires both explicit references; verifies current_user matches; rejects
superuser/create-role/create-db/replication/bypass-RLS, write/table/create privileges
and reachable write-capable role memberships before extraction. This SQL preflight
is code-reviewed and mock-tested but **PostgreSQL runtime UNPROVED**. It does not
prove every effective permission; operator security review remains mandatory.
Direct low-level acquire() is a bounded library helper, not a privileged-service
access boundary. Do not expose it to untrusted callers or use it to bypass CLI
preflight. No account data or financial ledger is queried.

## Bounds, snapshots and output

Explicit seven-symbol/five-timeframe registry; exact system environment; <=7-day
UTC receipt window, <=1000 rows/page, <=50000 rows, <=64 MiB source output, <=512 KiB
encoded row. SQL parameters are bound, no arbitrary query input. All pages/counts
use one read-only REPEATABLE READ transaction and C-collation event-ID pagination.
Statement/lock timeouts are transaction-local; rollback always ends acquisition.
Byte/row/ordering/scope failures never become complete exports. Max-row cap yields
partial status. Resume starts a **new** snapshot and is not falsely complete.

Destination must be absolute, new, private, with no traversal/symlink ancestors;
directory 0700/files 0600, exclusive creation. Partial write failures preserve
files for diagnosis and return a fixed failure code. Do not reuse those as complete
exports. Paths already writable by hostile users are not safe even with mode checks.
Private copies remain subject to access/retention and human review.

Timestamps serialize as canonical UTC fixed six digits with Z; payload original
strings remain original. Hash encoding is versioned v2 and incompatible with v1.
Original/redacted record and original-payload digests remain distinct. PostgreSQL
JSONB does not preserve exact original HTTP JSON bytes.

## Redaction and evidence

Structured, case-insensitive key matching covers token/secret/password/passphrase,
api_key/key/auth and connection/credential families, recursively in objects/arrays.
Known bearer/JWT/password-bearing DSN/connection strings are redacted. Legitimate
OHLCV/indicator numeric values are retained. Redaction paths do not echo detected
sensitive key text. No finite pattern list can detect every secret disguised as an
ordinary feature value: exports remain private, require human review and must not
be committed to GitHub. Only synthetic fixtures were used in this task.

Producer-generated manifest, file audit and record hashes are internally checked,
but do not establish authenticity. The independently retained expected TrustedAnchor
must pin manifest, audit, ordered IDs/digests/counts/scope/window/operator/version.
The producer tool cannot issue/approve a real anchor. Obtain it from a separately
approved custodian and pass it outside the export directory. See ANCHOR_SPEC.md.
Absent anchor means UNPROVED; --require-anchor fails. An expected-hash match is not
cryptographic proof that custody was independent. Custody/authentication policy is
an outstanding approval dependency.

Receipt-window completeness does not establish source-bar coverage: current rows
can have received_at replaced on retry, late records may fall outside the window,
and historical capture may be incomplete. Require source calendar/collection-start/
outage/catch-up evidence. No nominal 24-hour gap filling or rounding is permitted.

## First real acquisition prerequisites

Fresh independent code review; complete historical D1–D9 review package; disposable
PG18 runtime validation; CEO approval of dedicated identity/host/egress/retention;
independent expected-anchor custody; original logger/provider/version evidence and
approved session calendars. Then separately approve one stream/day pilot export.
Do not extract tonight and do not activate training or live features from this work.
