# Independent reproduction — source/evidence only

1. Full clone exact closureSHA; sole parent647c0f623b6c573d6e535600f34b39095f8a4c91. Full parent-to-candidate diff must add only operations/analytics_release_evidence_closure/. Old security, fallback and operations directories must be byte-identical. Verify SHA256SUMS and source inventory.
2. Python3.14.3 disposable runtime: seeded pip26.2.1 used as installer tooling, not runtime dependency. Download exactly15 matching binary wheels with hash checking into private scratch. Install with `python -m pip install --no-index --find-links WHEELHOUSE --require-hashes -r operations/analytics_release_evidence_closure/runtime-observed.lock`; run `python -m pip check`. Record actual Python/platform/ABI/package/installer inventory. Root production requirements remain unchanged.
3. Run verify_wheels.py with --wheelhouse,--lock,--output in disposable scratch. It hardlinks originals into a temporary directory, replaces one temporary copy with changed bytes and requires pip dry-run/ignore-installed/no-index hash rejection. It never alters source wheels or installs the changed package. File manifest identifies the selected platform wheels.
4. Separate test environment: use the previous operations hash lock at operations/analytics_containment_release/requirements-observed.lock (includes httpx and its test dependencies). Do not add test dependencies to runtime proposal.
5. From repository root, run new recorder twice with that interpreter and unique labels:

```
TEST_PYTHON operations/analytics_release_evidence_closure/record_tests.py TEST_PYTHON reviewer314a
```

It starts clean children with only PATH/PYTHONDONTWRITEBYTECODE. Expected16 new offline checks,42 original security tests,4 unchanged fallback tests,12 prior tooling tests and12 prior operations rehearsal tests:86 per run. Logs/commands/exit codes go only to the new closure namespace. Prior rehearsal fault injections intentionally log startup errors; inspect unittest result/exit codes, not ERROR words alone. No production variables, database, orders or public server are used. Full Git history is needed by baseline comparisons.
6. Validate source_inventory at immutable3467133 and5f0: fallback absent/present, runtime blobs identical; compare exact trees in ARTIFACT_STRATEGY.md. These are component facts, not release approval. Verify no record silently promotes a different merge result or unverified publication.
7. Read private consumer/observation/publication records through the assigned restricted destination. Validate manifest/ZIP SHA256, file hashes, item/parent ACLs and provider/source/session attribution limits. Do not copy originals to public engine GitHub. Unknown current owners/consumers remain unknown.
8. Current Render docs references/hashes are in evidence/documentation_references.json. Live queries, timestamps and actual source references belong in the private bundle. No live query is executed by this reproduction procedure. Independently verify remaining runtime, retention, consumers and recovery evidence through authorised operators only.

No author test, manifest, hash, proposal or successful reproduction grants CEO authority or overwrites independent assurance.
