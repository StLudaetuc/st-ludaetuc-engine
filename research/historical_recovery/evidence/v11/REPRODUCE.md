# Reproduction and evidence boundary

Full source parent: 2807fdde9d40d5084f85b9d4d0a3005cc50464f5.
Use a fresh isolated checkout at the exact new SHA in the handover. No production
DSN or environment values are needed. All tests use synthetic fixtures/mocks only.

Create Python 3.12 and 3.13 venvs and sync the test-only requirements.lock with
--require-hashes. Execute run_suite.py from the repository root with --python
pointing to each venv and --label naming a new evidence run. It records clean child
environment keys, source hashes, command, timestamps and exit code. Expected:
142 tests, 142 pass, 0 failure/error/skip, exit 0. Run twice per interpreter.

The original unmodified 99 reproduction can be rerun from a separate checkout of
2807fdde..., using the same locked test dependencies. Current original test files
remain unchanged; the new test_v11.py adds 43 regression methods.

Original99 logs are exact original runs. corrected_initial.log contains the preserved
development failure (three stale per-row digest errors), corrected_original99.log
is the subsequent passing 99 run. Earlier 134/140/142 development runs are labelled
first_full_regression / regression_with_privileges / pre_scope140 / pre_feature142;
they are not substitutes for final_* metadata/source hashes. No test log content was
manually edited. evidence/test_results.txt concatenates four final unedited logs.

sha256sum -c research/historical_recovery/evidence/v11/SHA256SUMS verifies changed
source and evidence except the self-referential manifest itself; Git binds its bytes.
Exact complete Git diff: git diff 2807fdde... <new-sha>. A full patch is included in
the downloadable engineering package after freeze. All paths must be beneath the
recovery namespace. No approved production/governance/frozen branch may move.

The complete reviewer package (including exact D5/D6/D8/D9 and its 38/51 checks) is
unavailable here. Do not claim it ran unchanged. Current results are author tests,
not independent assurance. PostgreSQL 18 is UNPROVED, not mocked PASS. See
PG18_REQUIRED_CHECKS.md for the outstanding genuine runtime matrix.

The runbook explicitly forbids the existing writer-bearing engine Web Shell.
Future acquisition requires independently approved SELECT-only identity/host,
reviewed source/calendars, external anchor custody, and separate CEO approval.
No actual market-data row, credential, financial record or production system was
accessed or changed by these tests. No deployment or extraction is authorised.
