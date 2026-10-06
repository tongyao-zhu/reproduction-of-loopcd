# Related implementations

Checked October 6, 2026 for **[Decoding Looped Transformers Better for (Almost) Free](https://arxiv.org/abs/2610.02185)**.

There are already public LoopCD implementations and experiments. In this search, we did not find another project reporting all eight Figure 1b comparisons. Our focus is that set of paired benchmarks.

| Project | What is available | Experiment scope |
|---|---|---|
| [apple-loopcd-off-the-shelf](https://github.com/tchayintr/apple-loopcd-off-the-shelf) | Python implementation, saved results and run scripts; checked at `e6ea740` | Qwen variants on ARC, MMLU and Thai tasks; Qwen and Ouro-1.4B/2.6B on 373 O-NET questions. Different model/task pairings from Figure 1b. |
| [vLLM-RLT LoopCD RFC](https://github.com/ThinkFlowLab/vllm-rlt/issues/85) and [Ouro PR #86](https://github.com/ThinkFlowLab/vllm-rlt/pull/86) | Ouro/Huginn prototype work, correctness checks and inference measurements; Ouro PR was open and draft at `d700ecc` | Small ARC-C pilots and 16-task HumanEval+ generation. The report says executable HumanEval scoring was not run. |
| [long-haul Parcae experiment](https://github.com/pH34r-pH/long-haul/issues/180) | A plan for Parcae-140M/370M quality and compute experiments | The issue's deliverables were unchecked, with no comments or completed results at the time of inspection. |
| [katgpt-rs](https://github.com/katopz/katgpt-rs) | Rust implementation of recurrent-depth guidance, with a reported negative result on its own test fixture | Its own model stack and fixture, rather than the paper's checkpoints and benchmarks. |

We checked general web search, GitHub repository and issue/PR search, the [first author's homepage](https://neosknight233.github.io/), and the [Hugging Face paper page](https://huggingface.co/papers/2610.02185). Queries included the exact title, arXiv ID, `LoopCD`, `reproduction`, `implementation`, and `复现`. GitHub's API surfaced projects that were missing from ordinary web-search results. We inspected the candidate repositories and their result files or issue status. We did not find the authors' complete implementation or evaluation configuration through those sources.

[byeonghoyu/LoopCD](https://github.com/byeonghoyu/LoopCD) belongs to a different paper, [arXiv:2609.24196](https://arxiv.org/abs/2609.24196), despite sharing the method name.
