# Figure 1b protocols and limitations

This project evaluates the same-depth comparisons in Figure 1b of arXiv:2610.02185v1. It does not reproduce the complete paper, training, all ablations, or all latency/compute claims. Parameters below were fixed before scoring; negative results are retained.

| Model / task | Depth | Early readout | Guidance | Evaluation |
|---|---:|---:|---|---|
| Ouro-2.6B Thinking / AIME 2024 | 4 | 1 | Adaptive cap 1.5 | 30 problems × 16 samples per arm; pending |
| Ouro-1.4B Thinking / AIME 2024 | 4 | 1 | Adaptive cap 1.0 | 30 problems × 16 samples per arm; pending |
| Huginn R32 / HumanEval | 32 | 7 | Hidden, ω = 0.3 | 164 problems per arm |
| Huginn R16 / HumanEval | 16 | 1 | Adaptive logits, cap 0.25 | 164 problems per arm |
| Ouro-2.6B / HumanEval | 4 | 1 | Adaptive logits, cap 1.0 | 164 problems per arm |
| Parcae-370M / ARC-C | 8 | 1 | Adaptive logits, cap 1.0 | 1,172 problems, 25-shot, acc_norm |
| Parcae-1.3B / HellaSwag | 8 | 1 | Adaptive logits, cap 1.0 | 10,042 problems, 0-shot, acc_norm |
| Looped-Qwen3 / MBPP | 8 | 1 | Fixed logits, ω = 0.3 | 378 problems per arm |

## Generation and scoring

Code generation uses greedy decoding, one solution per task and arm, a 2,048-new-token cap, and EvalPlus 0.3.1 chat-format prompts. These prompt/stopping choices are part of our reconstruction, not proof of the authors' original harness. HumanEval uses HumanEvalPlus-v0.1.10 (164 tasks); MBPP uses MbppPlus-v0.2.0 (378 tasks). Base-test pass@1 is reported in Figure 1b; extended-test results are retained separately in the statistical extracts. The dataset hashes are checked by the generators.

Huginn keeps its native truncated-Gaussian initialization; the per-task seed is reset for both arms. R16 adaptive guidance has its own reference coda cache. The previous R16 *hidden* experiment is not substituted for the Figure 1b R16 *logit* row. Ouro base and Thinking models have different native EOS conventions, which are preserved.

Multiple-choice scoring uses candidate continuation log-likelihood and reports acc_norm, with raw accuracy retained in the evidence. Prompt construction, few-shot requests, candidate boundaries, dataset identities and source hashes were audited before generation/scoring.

Generated programs are evaluated only in the project's Linux isolation runner after full canonical self-checks (164 HumanEval / 378 MBPP tasks). The EvalPlus 0.3.1 `find_zero` bookkeeping patch records successful progress; its residual criterion is unchanged. The preparation code records the patch diff and hash.

## AIME is still in progress

Both Thinking models use 8,192 new tokens, temperature 1, top-p 0.7, top-k disabled, 16 independently seeded samples per problem, and native EOS. The current backend is vLLM 0.13.0, BF16 eager mode, batch size 2, tensor/pipeline parallelism 1, prefix caching disabled. Each model is split across two GPUs by disjoint task ranges. Completed batches from the same vLLM protocol are retained with explicit provenance.

Earlier Hugging Face generations and throughput probes are not mixed into this result. BF16 kernels, process restarts, batching and backend changes can change sampled trajectories. Numerical and cache tests support the implementation; they do not prove that vLLM reproduces the authors' runtime or the earlier Hugging Face token sequences. No AIME score is reported until both arms contain all 480 validated samples.

## Looped-Qwen3 reconstruction

We use `Qwen/Qwen3-4B` at revision `1cfa9a7208912126459214e8b04321603b3df60c`. The authors' wrapper/checkpoint identity was not recovered. The implementation loops zero-based layers 15–18 eight times, updates `u_next = (7/8)u + (1/8)g(u)`, and reads h1 and h8 through the shared tail. It uses separate KV histories for logical depths and for the two tail branches. There is no added anchor or decode bypass.

This is a preregistered independent reconstruction, not a recovered author configuration. BF16 full-prefix versus incremental execution can differ through rounding; checks include an independent streaming reference and FP32 semantic comparisons. Matching a score approximately does not identify the authors' cache policy.

## What the public evidence establishes

Six comparisons have complete sample counts and paired checks. Five show a positive change; Huginn R32 does not reproduce the reported improvement. The public JSON files contain statistical extracts, original evidence hashes, and relevant source identifiers. They support inspecting the reported counts and arithmetic, but do not replace the full private raw-generation archive. The result verification command checks arithmetic and evidence consistency; it does not regenerate model outputs or independently execute benchmark tests.
