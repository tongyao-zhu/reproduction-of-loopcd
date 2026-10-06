# Huginn R32: what we checked

Our full HumanEval run scored **45/164 with and without LoopCD**. Guidance changed 136 of the 164 token sequences, fixed 11 answers and broke 11. It was active, but the gains and losses cancelled out.

The paper reports 22.56% → 31.71% for this row. Our baseline is 27.44%, so the starting point already differs. We have not identified a bug that explains the gap. The remaining configuration questions are listed in [experiment settings](protocols.md#details-we-chose).

## Does output length explain it?

We took the saved token sequences, cut them at a shorter budget, applied the same code cleanup and rescored the affected outputs. This checks the existing generation prefixes; it does not run the model again.

| Token budget | Baseline | Hidden LoopCD | Gain |
|---|---:|---:|---:|
| 2,048, original run | 45/164 · 27.44% | 45/164 · 27.44% | 0.00 pp |
| 768, saved prefixes | 45/164 · 27.44% | 45/164 · 27.44% | 0.00 pp |
| 256, saved prefixes | 42/164 · 25.61% | 44/164 · 26.83% | +1.22 pp |

At 256 tokens, the baseline loses three correct answers and LoopCD loses one. No previously wrong answer becomes correct. The positive difference comes from unequal losses under truncation. The 256-token baseline still exceeds the paper's by 3.05 points.

For 768 tokens, we rescored 26 long outputs; for 256, we rescored 64. The remaining outputs had identical cleaned code and retained their original verdicts. Both checks use the same isolated scorer, which had passed all 164 canonical solutions. Each final comparison includes all 164 tasks per arm.

The paper does not specify a token cap. These checks therefore stay separate from the main Figure 1b results, which retain the original 2,048-token run.

Per-task verdicts, counts and source hashes: [768-token check](../results/diagnostics/huginn-r32-768.json), [256-token check](../results/diagnostics/huginn-r32-256.json), [file hashes](../results/diagnostics/manifest.json).
