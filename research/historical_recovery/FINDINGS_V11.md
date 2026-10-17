# D1–D9 disposition — DRAFT; complete review package unavailable

The assignment identifies D1,D2,D3,D4,D7 by number. Neither the available repository
branches nor the bounded Library search located the full historical independent
review package. The reviewer-reported 38 offline/51 PostgreSQL checks were **not**
run unchanged. Do not invent the missing D5,D6,D8,D9 identities. Their exact review
wording/reproductions remain BLOCKED_ON_EVIDENCE; listed related lower-severity
requirements are separately addressed below without falsely assigning labels.

| ID | Original / reproduction / cause | Correction | Regression/result | Residual |
|---|---|---|---|---|
| D1 | Same datetime instant with UTC vs London offset hashes differently; reproduced in v1; serializer used session offset | UTC six-digit timestamp serialization and versioned digest | TimeV11 tests compare 601 synthetic comparisons x UTC/London/New_York; pass | Python tests, not genuine PostgreSQL timezone execution |
| D2 | All producer hashes/audit can be regenerated; a checksum cannot authenticate its producer | External TrustedAnchor pins manifest/audit/record population/count/scope/time/operator/version; audit files actually checked; missing anchor unproved or required-failure | AnchorV11 self-consistent altered export accepted only as UNPROVED, rejected against retained expectation; match/wrong/missing tests pass | Real independent custody/signing/source authenticity unproved; no trust issuer created |
| D3 | Invalid OHLCV occupies expected timestamp; v1 reports COMPLETE_EXPECTED_SLOTS; reproduced | Valid slots require final validated eligible observation; missing/invalid/conflict/unproved sets distinct | CoverageV11 invalid-slot negative and complete positive pass | Real approved calendar/source population unavailable |
| D4 | Conflicting payloads get zero eligible rows but recovery counts still ORIGINAL_TIMESTAMP_VERIFIED; reproduced | Conflict/duplicate demotion propagates to recovery aggregates; independent original timestamp state preserved; originals/features retained | Conflicting OHLCV/indicator tests pass; no unresolved conflict contributes to eligible slots | No automatic winner or correction policy |
| D5 | Exact finding unavailable | Not asserted resolved under this ID | BLOCKED_ON_EVIDENCE | Obtain full report |
| D6 | Exact finding unavailable | Not asserted resolved under this ID | BLOCKED_ON_EVIDENCE | Obtain full report |
| D7 | passphrase/api_key/key/auth bypass v1 patterns; reproduced | Case-normalized structural key matching, nested objects/arrays, bearer/DSN/password/JWT value checks, non-echoing sensitive paths | RedactionV11 positives/negatives pass | Generic arbitrary-value secrets cannot be universally recognised; private output/human review remain required |
| D8 | Exact finding unavailable | Not asserted resolved under this ID | BLOCKED_ON_EVIDENCE | Obtain full report |
| D9 | Exact finding unavailable | Not asserted resolved under this ID | BLOCKED_ON_EVIDENCE | Obtain full report |

Related requirements independently verified and corrected without invented D-labels:

- Floating/exponent milliseconds: strict lexical integral format, no float repair;
  seconds/date bounds retained. TimeV11 tests pass.
- Sub-microsecond ISO: reject precision loss explicitly, retain original string.
- Source versions: only known nonempty bounded strings; integer source version cannot
  gain acceptance via str() conversion; mixed declarations remain held.
- Scope: Context optional explicit instrument/timeframe, calendar stream bound even
  if calendar unapproved, export row scope checked against manifest and Context.
- Indicator names: case/period variants such as EMA20/RSI_14 recognised, source values
  copied without computation, field-level provenance and semantic-validity disclaimer.
- CLI privilege preflight: dedicated approved login, non-writer table/role/create/
  membership checks; named isolated execution context required. SQL runtime UNPROVED.
- Destinations: new absolute private path, no traversal/symlink ancestors, no overwrite.

Original 99-test reproduction passed unmodified on both interpreters. A development
iteration had three reader-digest errors after adding timestamp-state metadata;
fixed by recomputing each final projection digest. Its unedited corrected_initial.log
is retained; it is not a final passing result. All final results are separately
recorded. No missing reviewer run is represented as independently reproduced.
