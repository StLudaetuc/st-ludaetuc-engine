# AUTHOR EVIDENCE — full dependency comparison

Recommendation: **Set G for the engineering candidate**, subject to independent review. Neither set meets full release acceptance: PostgreSQL/original fallback-matrix evidence and pinned-container builds remain incomplete. RELEASE STATUS: NO-GO.

Rubric: (1) known security findings, (2) complete compatibility, (3) deterministic wheel availability, (4) operational change, (5) reviewer parity. Both sets have zero known findings in pip-audit 2.10.1 via PyPI and OSV, pass committed suites twice and supplemental mock-persistence trading checks. G preserves the verified original 15-wheel observed bundle and existing operations parity; R changes four primary packages plus pydantic-core and resolves opentelemetry-api 1.45.1 rather than G 1.45.0. R is a reconstructed current resolution, **not** byte-for-byte reproduction of an original reviewer lock. This provenance/impact difference favours G; it is not evidence that G is universally safer.

Runtime target: CPython 3.14.3, Linux x86_64; platform tags and Python markers/Requires-Dist are recorded for every package in G-wheels.json and R-wheels.json. Both offline installs and corrupt-wheel/lock rejection succeeded. Runtime packages are all wheels; no source-built dependencies were used. Test-only packages are hash-locked separately. pip itself is an installer, not an application runtime dependency.

| Package | G | R | G wheel SHA-256 | R wheel SHA-256 |
|---|---|---|---|---|
| annotated-doc | 0.0.5 | 0.0.5 | `117bac03a25ede5df5440e855b32d556049ca169ead221505badf432fed4b101` | `117bac03a25ede5df5440e855b32d556049ca169ead221505badf432fed4b101` |
| annotated-types | 0.8.0 | 0.8.0 | `f072f4d804ea359e4eaf198b1af7a8b0943881a87f31bb764f8bf219bb9419e0` | `f072f4d804ea359e4eaf198b1af7a8b0943881a87f31bb764f8bf219bb9419e0` |
| anyio | 4.15.1 | 4.15.1 | `6152fdbbf9a77fdec97731721bebf7c4c44f7c29b424b0065826173efc7ed101` | `6152fdbbf9a77fdec97731721bebf7c4c44f7c29b424b0065826173efc7ed101` |
| click | 8.5.0 | 8.5.0 | `255bc9599cf7748b4b1a446ccc735421bd08a2ae529a8b88597d3de5664ee360` | `255bc9599cf7748b4b1a446ccc735421bd08a2ae529a8b88597d3de5664ee360` |
| fastapi | 0.142.2 | 0.143.3 | `bd5f4d81f1e93a88bcd77caf4dfe3c2dbffc3805407a0007e9a114c18b3a670b` | `75e8bd68e2f52f77d29b2879749c257d64de2d85282d25270caf2bfccce5efe5` |
| h11 | 0.16.0 | 0.16.0 | `63cf8bbe7522de3bf65932fda1d9c2772064ffb3dae62d55932da54b31cb6c86` | `63cf8bbe7522de3bf65932fda1d9c2772064ffb3dae62d55932da54b31cb6c86` |
| idna | 3.20 | 3.20 | `ab7ae7122974553370f0bdb919e1a960b2cd1bc1ef0276416d896db81c14582c` | `ab7ae7122974553370f0bdb919e1a960b2cd1bc1ef0276416d896db81c14582c` |
| opentelemetry-api | 1.45.0 | 1.45.1 | `80e068aba7cd56c8b58512d6a36f8d25cb1dfaa0c0a4cc1c938ccf9f362d9cb3` | `b31553efa588ae44bc306f863c785c5333a9ecc091248c6ee68b4b6c87fdedfb` |
| psycopg2-binary | 2.9.13 | 2.9.13 | `190c18b97d9ef72f2e88c451b6588af90d6bd7bf54cb94b963280dc86a2c7076` | `190c18b97d9ef72f2e88c451b6588af90d6bd7bf54cb94b963280dc86a2c7076` |
| pydantic | 2.13.5 | 2.14.1 | `346a034f080da3755d8e9cb5e00e8b07de1d39e4f6e2c87d8ab7cafa0b269a73` | `9195d967ec791692a04438115466764fb8b9a27b31f14a760437694f40d6b454` |
| pydantic-core | 2.46.5 | 2.50.1 | `54d510bac3ee52247af28ed4bb18a1e799f040ac60fd2bf5ccd4c92f1fbe786f` | `42b54c2c90ad348b5e3a85e03e715d572c1fde357ef104cdfe3b03b697a404ea` |
| starlette | 1.7.0 | 1.8.0 | `67f8e99895493dd2911a03f11314af6ceebeae4e704bb9f43dfc6a9db151c93e` | `dfdd6b29c26483288088d990eee59631dedadd66ce20d203402a7ca8e3c4656f` |
| typing-extensions | 4.16.0 | 4.16.0 | `481caa481374e813c1b176ada14e97f1f67a4539ce9cfeb3f350d78d6370c2e8` | `481caa481374e813c1b176ada14e97f1f67a4539ce9cfeb3f350d78d6370c2e8` |
| typing-inspection | 0.4.4 | 0.4.4 | `65b8397ba37ccbce054456aaccddfc91e6e3083c92824df348d96ca832f3f147` | `65b8397ba37ccbce054456aaccddfc91e6e3083c92824df348d96ca832f3f147` |
| uvicorn | 0.54.0 | 0.54.0 | `505bdb0f318731d45f1f712071fc781a8981f6847a31c902c9f5e652d4f67faf` | `505bdb0f318731d45f1f712071fc781a8981f6847a31c902c9f5e652d4f67faf` |

All package filenames, hashes, full transitive dependencies, environment markers and wheel tags are machine-readable in the accompanying inventories. Advisory result/triage: zero returned findings in both 15-package scans through both services; there are no returned high/critical advisories to triage. This is bounded to these sources at retrieval, not proof of no undisclosed vulnerability. Neither service exposes a complete database snapshot version/timestamp in these responses; database timestamp = NOT_EXPOSED. Preserve request execution timestamps and raw JSON, not an invented DB version.

| Check | G | R |
|---|---|---|
| Committed primary/fallback/tools/rehearsal/closure + C3 | 91/91 ×2 | 91/91 ×2 |
| Original v1.1.1 root harness | candidate 0/20; baseline 20/20; v1 14/20 | candidate 0/20; baseline 20/20; v1 14/20 |
| Supplemental trading differential | 236 requests, zero response differences, equal in-memory event/schema hashes | same |
| Real PostgreSQL persistence | INCOMPLETE | INCOMPLETE |
| Original fallback matrix | UNAVAILABLE | UNAVAILABLE |
| pip check / hash rejection / offline install | clean / rejected / installed | clean / rejected / installed |
| Two pinned-container reproducible builds | INCOMPLETE | INCOMPLETE |

Original inputs: G lock is the exact base operations lock and recovered original wheelhouse. release/G.lock narrows it to the exact 15 selected platform wheels, preserving all observed versions and transitive requirements. R is resolved from the five specified versions using binary wheels, with full hashes generated from acquired bytes; without the original reviewer archive/lock, transitive parity cannot be asserted. The production environment was not read. A Render PYTHON_VERSION variable overrides .python-version and requires later operator verification.

Original reviewer security harness is reproduced unchanged; its synthetic pool is not PostgreSQL. Author root control v1 composes the original v1 middleware with baseline engine; its engine AST difference is the pre-existing import/middleware addition, not a production target. Supplemental 236-call fixture is explicitly reconstructed from prior author synthetic payloads and is NOT substituted for the unavailable reviewer fixture.

Wheel provenance: G bundle checksum verified before extraction. Current index/attestation requests and any access limitations are retained in restricted wheel-provenance.json. No source-build flags are inferred from merely installed metadata. PyPI attestations, when unavailable, are not a pass. Hash locks prove downloaded byte identity against chosen hashes, not upstream trust.

Current PyPI metadata independently matched all 30 selected G/R wheel filenames and SHA-256s. Provenance endpoints returned 200 for 16 artifacts and 404 for 14; returned records are retained. Retrieval does not by itself verify attestation signatures or confer independent approval.
