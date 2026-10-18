# AUTHOR EVIDENCE — first release protection, A25 NOT MET

RELEASE STATUS: NO-GO. The retained unprotected baseline is an exposure path until a separately established boundary covers it. Synthetic ASGI results are design evidence, not Render enforcement.

| Option | Isolated result | Later operator / Production actions required |
|---|---|---|
| Preverified immutable artefact | Reduces build/start uncertainty; cannot affect requests to a retained old process when the deployment fails | Authorised immutable-image configuration, exact digest verification, staging failure rehearsal and a separate protective boundary before the first rollout |
| Minimal-change R-B first release | Contains analytics once it actually serves; if first deployment fails and old instance remains, that old instance still leaks | Separately reviewed recovery composition, operator-approved first deployment and proven removal/protection of all baseline paths; this is not achieved by changing the start command |
| Independent deny ingress | Model blocks analytics in all phases despite failed candidate build/start; direct-origin positive control still leaks | Operator establishes an independently controlled deny proxy before release, makes origin inaccessible except through that proxy, verifies default provider hostname/custom domains/alternate ports/normalisation/WebSocket paths cannot bypass it, verifies trading availability and audit retention, then fault-tests failures. Requires separately authorised network/service changes |

`python -m release.first_release_demo` models an old instance remaining live before/during/after failed build/start. It exercises four canonical analytics routes, case/slash/newline variants and a direct-origin positive control. All requests through the model barrier are denied; bypass data remains accessible. It does not cover all real HTTP proxy transformations or establish origin isolation. Therefore the all-path requirement is NOT MET and A25 cannot pass.

A planned outage and post-release rollback do not prove first-release containment. No ingress, Render or Production action was performed or authorised here.
