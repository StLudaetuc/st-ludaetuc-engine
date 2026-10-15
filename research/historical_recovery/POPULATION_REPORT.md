# Historical population recovery assessment — DRAFT

## Source scope

The CEO supplied a read-only PostgreSQL population summary in the assignment.
It was not independently re-queried by this implementation. Exact SQL, capture time,
row manifests, overlap count and per-stream breakdown were not supplied.

| Reported observation | Count / state |
|---|---:|
| Stored market_data events in captured population | 164,209 |
| ISO payload.time format | 121,309 |
| Unix-millisecond payload.time_open format | 42,900 |
| Records lacking both fields | 0 |
| Reported streams with a source field | 35 |
| Real raw rows obtained/validated in this task | 0 |
| Valid source-bar timestamps | UNKNOWN |
| Invalid source timestamps | UNKNOWN |
| Missing expected bars | UNKNOWN |
| Duplicate/conflicting source bars | UNKNOWN |
| Population complete since capture inception | UNPROVED |

The reported source fields materially improve the recovery outlook: both format
families have original time candidates. They do not prove calendar alignment,
timezone/unit correctness, provider lineage, confirmed closes or coverage of every
expected bar. No trustworthy recovered-record count can yet be assigned.

The earlier 21,144-row export audit covered a different bounded day and omitted
original payloads. Its 35-stream off-grid stored timestamps are not a count of
corrupt source times and must not be projected onto this larger snapshot. The
old timestamp-fallback explanation remains a hypothesis for historical records.
Source inspection does establish that current ingestion reads body.time and does
not use body.time_open: actual retained raw payloads must establish format-change
boundaries and which stored event_times disagree. Do not rewrite either field.

The 35-row matrix in evidence/population_summary.json uses the prior observed
seven-symbol/five-timeframe roster as a proposed acquisition registry. CEO reports
all 35 have source fields; exact new snapshot roster/counts still need confirmation.
Every stream is NOT_YET_ASSESSED at record level, invalid/missing counts are null,
and research readiness is BLOCKED_ON_ROW_LEVEL_EVIDENCE. No record is called
UNRECOVERABLE just because its raw source has not yet been exported.

## Full-population verification proposal

1. Reproduce the CEO query in an authorised read-only snapshot. Retain exact query,
   capture time, total/format/overlap counts and per-stream/per-version counts.
2. Acquire bounded, private, redacted samples with original envelope/payloads from
   all streams, multiple times/day, session breaks, script and format boundaries.
3. Ratify source-version/field mappings and independent session calendars. Establish
   provider/ticker identity, confirmed-bar semantics and millisecond units.
4. Validate samples without rounding/replacing source facts. Bind reviewed exact
   digests to provenance references, keeping source/redacted representations distinct.
5. Authorise exhaustive bounded exports with snapshots/manifests, byte caps and
   coverage plans including late receipts/retries. Source-window completeness cannot
   be established from received_at alone. Add capture start/outage/expected-session
   evidence and a reviewed catch-up watermark before asserting coverage.
6. Audit each explicit stream/window, retain conflicts and malformed exceptions,
   then compare populations and expected slots. Only publish aggregate quality
   summaries and approved allowlisted reader products to authorised research users.
7. Independent review before production integration, training or backtesting.

Nothing in this proposal repairs production records or creates a competing ledger.
PostgreSQL remains the original-event authority. No fixture calendar is policy.
