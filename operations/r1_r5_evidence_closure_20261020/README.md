# R1–R5 supplemental evidence closure — Stage 0

**SUPPLEMENTAL AUTHOR EVIDENCE — not independent approval. RELEASE STATUS: NO-GO.** Prior verdicts remain unchanged. A25 remains NO-GO and was not worked on.

Candidate `d7d9b9c9d55027806d9f5f6cc8d20ee8a456ba67` (tree `89c51f474652c5eed22f4608c39d5f1a308276d8`) and R-B `8c3a7ca2a5f767ef49cbd8cd1cd5023a97de9e9c` (tree `312f3e2b4e12d02f7ae9f4d61a25fefdd1a1b356`) are unchanged. Only evidence scripts/documentation are added on the separate evidence branch. No Render, Production, deploy, merge, credentials, broker/live-trading or Chief action.

## New outcomes

- Recovered the ORIGINAL fallback archive through existing private GitHub read access. SHA-256 `4a8a89f810366d389303d78cc91bc50b588d4af4e45ab1c666cca99eacb14d8e`, 3,563,408 bytes, internal 131/131 checks. Original reviewer findings and code are retained unchanged; reruns are author evidence, not independent review.
- Original adversarial assertions: 13/13 ×2 on each G/R set. No DATABASE_URL configured: handler/router/lifespan assertions only, not the PG/SELECT matrix. Original v1.1.1 candidate root-path harness: 0/20 exposed on G and R; baseline/v1 controls recorded separately.
- Original trading fixture source is now available. Its 236-call corpus includes two non-trading paths. The disclosed supplemental trading-only 236 uses 234 original calls plus repeats of original indices0 and1; all eight real HTTP modes have zero response differences. No stubbed log_event or SQL is substituted, but the DB is unconfigured: no persistence claim.
- Actual loopback TCP refusal gives health200 and market-data500 in primary/fallback/RB. This is genuine connection failure with unchanged source, not mocked _get_pool. It is not a PG18 integration test.
- Verified Gitleaks8.30.1 and complete repository history/bundle now available. Raw findings remain visible and triaged; scanner exit1 is retained. Do not call this a zero-finding scan.
- Fresh Python3.14.3 offline G/R installs, pip-check, tampered wheel/lock rejection, all15 G SBOM/wheel checks and exact repeat of original host artifact digests completed. No pinned container was run.

## Precise infrastructure blockers

No docker/podman executable or daemon socket is available. No verified image digest was executed. PG18.0 source checksum was verified, but configure lacks bison; more decisively, the execution namespace maps only UID0. Switching to nobody fails (setresuid Invalid argument), and nested UID mapping fails (Operation not permitted). PostgreSQL’s non-root safety check was not patched or bypassed. Therefore actual schema creation, event persistence, row/schema/view hashing, SQL-positive controls and PG recovery drills are NOT EXECUTED. The guarded PG18 original-matrix adapter is supplied but unexecuted. A failed prerequisite is INCOMPLETE, not an application failure or PASS.

## Evidence-closure matrix

| ID | Previously incomplete | New author evidence | Current scope/status |
|---|---|---|---|
| A4 | Original fallback archive/matrix unavailable; no real PG | Original archive recovered and SHA-256 verified; 131 checks pass. Original 13-test handler/router/sole-barrier suite 13/13 ×2 on G/R (52 executions); real WS transport 24 denials; original PG matrix source available unchanged. | PARTIAL — full original real-Uvicorn/SELECT matrix NOT EXECUTED without PG18 |
| A12 | Pinned-container builds unavailable | Fresh CPython3.14.3 G/R offline installs; tampered wheel/lock rejected; G SBOM matches all 15 selected wheels; two fresh host builds per frozen source exactly reproduce both original artifact hashes. | INCOMPLETE — no container runtime/image execution or pinned-image provenance |
| A13 | Original 236 fixture unavailable; mock events only | Original corpus source recovered. Discovered original 236 includes /nonexistent and /openapi.json: 234 original trading requests + two disclosed author repeats = supplemental trading-only 236. 8 real HTTP targets (G/R × baseline/primary/fallback/RB), zero status/header/body differences, only Date removed. | PARTIAL — actual PG row/schema/view comparisons NOT EXECUTED |
| A15 | No disposable PG recovery drill | Original real PG differential/recovery-related failure logs recovered for provenance; real no-server TCP failure probes use unchanged candidate/fallback/RB. | INCOMPLETE — no new PG-backed startup-failure restore drill |
| A16 | Shared failure/persistence continuity unproved | Actual loopback TCP refusal (no stubs) yields /health 200 and /market-data 500 in candidate, fallback and RB. Recovery HTTP differential matches without DB. | PARTIAL — health/persistence blind spot demonstrated; PG continuity/recovery still INCOMPLETE |
| A18 | Gitleaks absent, local shallow history | Official Gitleaks8.30.1 binary digest verified. Complete public mirror and verified bundle, all refs captured. Initial snapshot 61 reachable commits; scanner reports 57 with patches. 23 findings triaged provenance hashes/test JWT; exact original author archive 117 provenance-hash findings and original fallback archive 1 RFC6455 nonce finding; no unresolved within these contract scans. | SUPPORTED BY NEW AUTHOR EVIDENCE — raw scans contain findings; not a zero-finding/independent PASS |
| A20 | Author archive byte-verified; reviewer access unknown | Original 26,117,666-byte author archive rehashed against expected hash. Current Drive ACL remains owner-only. Read-only original fallback retrieval method recovered; no new tokens or ACL edits. | INCOMPLETE — independent reviewer retrieval/access NOT VERIFIED |
| A21 | Restricted originals/access and operator evidence OPEN | Recovered originals retain canonical private assurance source IDs and hashes. Exact-copy originals are supplied for review in supplemental evidence without altering their canonical records. Operator requirements remain OPEN. | PARTIAL — distinct access and operator evidence remain OPEN |
| A22 | Both sets tested incompletely; original R parity unproved | All 15 R runtime versions match recovered reviewer freeze (Python3.14.6 original vs3.14.3 tested). Original hash-lock not present. Fresh hash-enforced installs/checks/tamper tests, unit/HTTP/WS comparisons on both sets; G selection unchanged. | PARTIAL — no PG18 or pinned-container builds for either set |
| A24 | Separate R-B existed, PG/pinned builds incomplete | Frozen RB reverified unchanged; exact original host artifact reproduced twice; real HTTP/WS and connectivity-refusal observations included. | PARTIAL — known-good PG recovery/pinned build acceptance remains INCOMPLETE |

The machine-readable matrix lists original supporting artifacts and the exact independent-review obligations for every row. No previous A-row verdict is changed retroactively.

## Exact commands and interpretation

Commands actually run include:

```sh
uv python install 3.14.3
python3 supplemental-20261020/scripts/run_original_units.py
envG/bin/python supplemental-20261020/scripts/original_fixture_http.py
python3 supplemental-20261020/scripts/repro_checks.py
python3 supplemental-20261020/scripts/host_rebuild.py
python3 supplemental-20261020/scripts/persistence_failure_probe.py
python3 supplemental-20261020/scripts/websocket_transport_probe.py
env -i PATH=/usr/bin:/bin supplemental-20261020/tooling/gitleaks/gitleaks git supplemental-20261020/history.git --log-opts="--all --full-history --diff-merges=first-parent" --redact --ignore-gitleaks-allow --max-archive-depth 4 --report-format json --report-path supplemental-20261020/logs/gitleaks-full-history-final.json
```

Raw logs include the actual absolute paths and complete environment inventories. Original Python3.12 mislaunch of the fixture adapter is retained as setup error and not counted. Test-only websockets17.2 was hash-installed separately for transport probes; frozen runtime/test locks were not edited. Scripts assume the supplied workspace layout: frozen checkouts, verified original harness directory, exact wheelhouses and disposable output directories. Public script copies bind the tested bytes; materialise the archive layout before rerunning.

Host artifact hashes reproduce the originals: candidate `0801321dd167106f0511306a127f1efe9a2a194fe02aeff222e417853f695cad`; R-B `0b3df73c3eac2a30fa1e459839aa01962f8ee48312186bccb52270a158aef209`. These are packaging-repeatability results, not pinned-container or Production acceptance. Executing frozen G packaging from a separate R tooling venv does not create an R application image.

## Access and next independent review

Author download remains different from reviewer access. Existing owner-only ACLs are unchanged. The original author archive remains at https://drive.google.com/file/d/1BfFKE0id12S00NpFOk84S_eM5pFUXt-V/view with SHA-256 `3da7f9dd1ca5a9f875abbce6bd7a94dd3b7efd77b41cf9bafd079cdaf210b605`. No independent actor download has been observed. Keep EVIDENCE_PUBLICATION_BLOCKED for this requirement. See ACCESS_HANDOFF.md for read-only retrieval without credential minting/public sharing.

Remaining mandatory blockers: actual disposable PG18; pinned-container builds/provenance; full recovery continuity; independent custody/reviewer access; all operator-owned evidence. A25 remains outside scope and NO-GO. Supplemental publication receipts bind new exact file IDs/hashes after upload; do not confuse these with earlier canonical archives.

No merge, deploy or Production action is requested or authorised by this handover.

Author report UTC: 2026-10-20T07:34:01.156274+00:00. Session ID NOT_EXPOSED; workspace UUID is not substituted. Provider OpenAI ChatGPT Work / Codex; single owner account StLudaetuc.
