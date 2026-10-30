# Batch 3 — out-of-sample check of a batch-1 lead (pre-registered 31 Oct 2026, before testing)

Batch 1 idea B failed overall, but two commodity pairs (Brent/WTI, corn/wheat) were positive in every period. They were
noticed after the fact, so they prove nothing. Hypothesis: **spreads between commodities in the same complex
mean-revert** (substitution and processing links pull them together). Honest test: the same rule (idea B exactly:
60-day z-score, enter |z|>2, exit |z|<0.5 or 20 days, 1.5× spread both legs) on same-complex pairs **not tested before**:
soybeans/corn, wheat/soybeans, sugar/corn, silver/platinum, gold/platinum, copper/palladium, natural gas/WTI.
PASS (pooled): mean > 0 in dev, val and holdout; date-clustered t ≥ 2.5 over the full sample; PF ≥ 1.2 (1.5× costs) and
≥ 1.05 (3× costs). If it fails, the batch-1 pairs are treated as chance.
