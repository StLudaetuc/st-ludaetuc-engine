# Release artifact strategy — PROPOSAL, not a release

## Immutable evidence versus deployable approval

| Exact reference | Components | Decision |
|---|---|---|
| `3467133b2208a09e1289651b5be0b45342695d52`, tree `bd5a011d40d4c6ecee541a7a4649ed154f83fc61` | Reviewed primary engine/middleware, no fallback files | Do not activate the fallback command on this tree |
| `5f0d07565629fba30d0c2221229fd2483472e701`, tree `df1f91d8d75a1a6fb204277e33eb01130af82c38` | Same engine/middleware plus wrapper, entrypoint, four fallback tests and old runbook | Exact fallback-bearing source reference; independent operational verdict REQUIRES_CORRECTION; not approved for production |
| `647c0f623b6c573d6e535600f34b39095f8a4c91` | Prior operations evidence | Not a runtime release target |
| This closure candidate | Operations documentation, offline verification and proposals only | Not a runtime release target |

**There is no approved deployable release SHA.** It would be false to fill that field with 5f0 merely because its component inventory is complete. The proposed final runtime artifact requires separately scoped engineering/approval to integrate accepted operational corrections and a reproducible dependency/runtime build, then exact-SHA independent review. This assignment creates none of those changes.

Proposed artifact bill of materials: unchanged reviewed engine and primary middleware; exact frozen outer fallback files if accepted for defence-in-depth; accepted runbook revision; approved Python/build constraints; complete hash-locked direct/transitive requirements and platform wheel manifest; build provenance/attestation; tests and independent review references. Runtime dependency proposal is separate from reviewer-resolved later package versions. No private records, env files, tokens or evidence datasets belong in the artifact.

## GitHub history and protected refs

Only the closure branch is writable. Main, all security candidates, previous operations, reviewer branches and Chief resources remain untouched. Read-only ancestry/tree tests do not prove current branch protection or signed approvals; those need repository-operator verification. No force-push, history rewrite, protection bypass or automatic merge is proposed.

A final source SHA must be selected and reviewed first. Immediately before approval, verify current main, clean worktree, exact parents, full main-to-target diff, source tree, component blobs, absence of unrelated work and final artifact manifest.

- Fast-forward: only if protected-branch policy explicitly permits it, main is an ancestor, and CEO approval binds the exact target SHA/tree. Main would become that exact SHA and Render on-commit auto-deploy would start. Do not assume fast-forward permission from ancestry.
- Merge commit: resulting SHA differs from source SHA. Prepare the prospective result in isolation before main changes; record both parent SHAs, result tree and result SHA. Independently review that immutable result. Freeze concurrent main changes. If GitHub would generate a different SHA, approval must explicitly bind the already-reviewed source tree plus exact parent set and an authorised resulting-commit verification mechanism **before** the ref change, or hold. Do not discover the mismatch only after go-live. The conservative default is exact result SHA known in advance.
- Squash/rebase/merge queue: also creates different identities and may change tree/parents. Same pre-main review requirements; no blanket equivalence by branch name. A changed base or merge result invalidates approval.

After an approved future main update, Render auto-deploys the actual main commit. Do not also trigger deployment manually. Operator records actual deploy ID and SHA, obtains that Git object read-only, computes its tree and component blobs, and compares all to the approved record. Platform live status is not behavioral acceptance. A drifted result is an incident; it must not be rationalized as equivalent after the fact.

## Build identity

Git SHA alone does not fix dependencies under `pip install -r requirements.txt`. Proposed version record includes Python3.14.3 build, OS/architecture/libc, all15 observed runtime packages, selected wheel SHA256s, installer version, source SHA/tree, lock SHA256, build/start commands, build ID/time and logs. Test-only dependencies remain separate. Offline wheel replay is engineering evidence, not proof of the next native Render build.

For native Render, a future separately approved change could install the hash lock and pin Python, with a retained wheel bundle or controlled index. Stronger isolation is an independently reviewed immutable image digest containing runtime/wheels, but it requires a separately approved service/build change. Neither is configured here. Any dependency set outside the reviewed artifact requires fresh testing and assurance.

## Separate recovery artifact

The current outer wrapper shares engine and dependencies; it is additional containment, not recovery from their failure. A genuine alternative would be an independently built, hash-locked last-known-good trading engine **composed with an unconditional outer denial barrier as its mandatory entrypoint**, plus dedicated acceptance and actual database-compatibility tests in disposable environments. Bare baseline and vulnerable v1 are prohibited. No such independently accepted recovery artifact SHA/digest exists today. The proposal does not certify that a composition automatically inherits either component's assurance.

Build and review it separately under new CEO scope, using no production data/credentials. Prove sole-barrier denial including proxy/root/encoding variants, no SQL reads on analytics, trading response/persistence equivalence, import/lifespan/dependency failure behavior, audit/access-log preservation, package-shadowing resistance and exact launch command. Validate artifact availability and all entrypoints before release approval. Independent testing plus CEO approval is required before it becomes a recovery option.
