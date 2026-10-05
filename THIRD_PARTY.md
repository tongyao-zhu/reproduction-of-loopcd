# Third-party resources

This is an independent implementation of the method described in [arXiv:2610.02185](https://arxiv.org/abs/2610.02185). Apple did not author or endorse this repository.

The code loads separately distributed models from ByteDance/Ouro, tomg-group-umd/huginn-0125, SandyResearch/Parcae, and Qwen/Qwen3-4B. Model preparation scripts pin revisions and describe compatibility patches. Their weights, tokenizer assets, remote model code and licenses are not bundled here.

Evaluation relies on separately installed Transformers, PyTorch, vLLM, lm-evaluation-harness and EvalPlus. Benchmark data is acquired from its original distributors. These dependencies and datasets retain their own licenses. Our original code's Apache-2.0 license does not relicense them.

The isolated EvalPlus setup records a two-assignment bookkeeping patch to the 0.3.1 `find_zero` success branch. It retains the numerical acceptance criterion and saves the original source, diff and hashes in the private evaluator runtime. No third-party runtime or generated benchmark programs are included in this export.
