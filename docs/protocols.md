# Experiment settings

We ran the eight same-depth comparisons in [Figure 1b](https://arxiv.org/html/2610.02185v1), using the methods and guidance parameters below. This is the scope of the experiments in this repository. Settings were fixed before scoring.

| Model / task | Depth | Early readout | Guidance | Evaluation |
|---|---:|---:|---|---|
| Ouro-2.6B Thinking / AIME 2024 | 4 | 1 | Adaptive cap 1.5 | 30 problems × 16 samples per arm; complete |
| Ouro-1.4B Thinking / AIME 2024 | 4 | 1 | Adaptive cap 1.0 | 30 problems × 16 samples per arm; complete |
| Huginn R32 / HumanEval | 32 | 7 | Hidden, ω = 0.3 | 164 problems per arm |
| Huginn R16 / HumanEval | 16 | 1 | Adaptive logits, cap 0.25 | 164 problems per arm |
| Ouro-2.6B / HumanEval | 4 | 1 | Adaptive logits, cap 1.0 | 164 problems per arm |
| Parcae-370M / ARC-C | 8 | 1 | Adaptive logits, cap 1.0 | 1,172 problems, 25-shot, acc_norm |
| Parcae-1.3B / HellaSwag | 8 | 1 | Adaptive logits, cap 1.0 | 10,042 problems, 0-shot, acc_norm |
| Looped-Qwen3 / MBPP | 8 | 1 | Fixed logits, ω = 0.3 | 378 problems per arm |

## Generation and scoring

Code generation uses greedy decoding, one solution per task and arm, a 2,048-new-token cap, and EvalPlus 0.3.1 chat-format prompts. The paper specifies zero-shot chat instructions and greedy decoding; the 2,048-token cap and exact prompt/stopping version are our choices. HumanEval uses HumanEvalPlus-v0.1.10 (164 tasks); MBPP uses MbppPlus-v0.2.0 (378 tasks). Base-test pass@1 is reported in Figure 1b; extended-test results are retained separately in the statistical extracts. The dataset hashes are checked by the generators.

Huginn keeps its native truncated-Gaussian initialization; the per-task seed is reset for both arms. R16 adaptive guidance has its own reference coda cache. The previous R16 *hidden* experiment is not substituted for the Figure 1b R16 *logit* row. Ouro base and Thinking models have different native EOS conventions, which are preserved.

Multiple-choice scoring uses candidate continuation log-likelihood and reports acc_norm, with raw accuracy retained in the evidence. Prompt construction, few-shot requests, candidate boundaries, dataset identities and source hashes were audited before generation/scoring.

Generated programs are evaluated only in the project's Linux isolation runner after full canonical self-checks (164 HumanEval / 378 MBPP tasks). The EvalPlus 0.3.1 `find_zero` bookkeeping patch records successful progress; its residual criterion is unchanged. The preparation code records the patch diff and hash.

## AIME

The paper does not give a generation-token cap. We chose 8,192 tokens and used temperature 1 / top-p 0.7 from the Ouro paper. The full prompt, answer extraction, seed schedule and backend were fixed for our runs; the authors' corresponding settings are not available in the materials we found.

Both Thinking models use 8,192 new tokens, temperature 1, top-p 0.7, top-k disabled, 16 independently seeded samples per problem, and native EOS. The current backend is vLLM 0.13.0, BF16 eager mode, batch size 2, tensor/pipeline parallelism 1, prefix caching disabled. Both models are complete. Ouro-2.6B used five disjoint worklists across two machines. Completed batches from the same vLLM protocol are retained with explicit provenance.

Earlier Hugging Face generations and throughput probes are not mixed into this result. BF16 kernels, process restarts, batching and backend changes can change sampled trajectories. Numerical and cache tests support the implementation; they do not prove that vLLM reproduces the authors' runtime or the earlier Hugging Face token sequences. A score is reported only after both arms contain all 480 validated samples. Ouro-1.4B passed this check and an independent local re-score: 168/480 → 195/480, or 35.00% → 40.625%. Its paired problem-bootstrap 95% interval for the gain is [0.83, 10.83] percentage points (10,000 resamples, seed 42, keeping all 16 samples of a problem together). Ouro-2.6B also passed the full independent re-score: 211/480 → 254/480 (43.9583% → 52.9167%), a gain of 8.9583 points with a [4.79, 13.54] problem-bootstrap 95% interval. Neither interval accounts for protocol differences.

## Looped-Qwen3 reconstruction

We use `Qwen/Qwen3-4B` at revision `1cfa9a7208912126459214e8b04321603b3df60c`. We could not find the authors' exact checkpoint and wrapper, so we built this version from the paper: loop zero-based layers 15–18 eight times, update `u_next = (7/8)u + (1/8)g(u)`, and read h1 and h8 through the shared tail. Each logical depth and each tail branch has its own KV history. We add no anchor or decode bypass.

These choices were recorded before generation. We checked them against a separate streaming reference and FP32 calculations. The authors' cache policy remains unknown.

## Details we chose

The paper gives the guidance formulas, depths and strengths. It leaves several execution details open. These are the settings used here, rather than settings confirmed from author code:

| Detail | Our setting |
|---|---|
| Huginn randomness | Native truncated-Gaussian initialization; seed 42 plus a task-ID hash, reset for each arm and task; batch size 1. Greedy token selection still depends on the random initial state. |
| Huginn hidden readout | Combine the recurrent block outputs before the first `ln_f`, then use the native coda and head. Apply guidance to all prefill positions and keep the guided coda history. |
| Code prompt and stops | EvalPlus 0.3.1 chat instruction, checkpoint chat template, that version's stop strings and the model's native EOS. |
| Runtime | BF16 model weights and FP32 guidance, PyTorch 2.9.0 / Transformers 4.54.1 for the Hugging Face runs; the AIME vLLM settings are listed above. |
| Parcae evaluation | Candidate continuation likelihood with `acc_norm`; raw accuracy is also saved. The paper's exact normalization and few-shot example selection are unconfirmed. |

Huginn weights are pinned to `tomg-group-umd/huginn-0125@bb6621b65e90b6a4b9b29ef88dc83866d450470c`. A cache-interface compatibility patch is applied to a local model copy. The adapter, prompt, cache, seed and scorer checks are recorded, but the authors' complete configuration would still be needed to compare these choices directly. The [Huginn notes](huginn.md) cover the baseline and length checks.

## What the public evidence establishes

All eight comparisons have complete sample counts and paired checks. Seven show a positive change; Huginn R32 does not reproduce the reported improvement. The public JSON files contain statistical extracts, original evidence hashes, and relevant source identifiers. They support inspecting the reported counts and arithmetic, but do not replace the full private raw-generation archive. The result verification command checks arithmetic and evidence consistency; it does not regenerate model outputs or independently execute benchmark tests.
