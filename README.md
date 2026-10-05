# Reproduction of LoopCD

An independent reproduction of **[Decoding Looped Transformers Better for (Almost) Free](https://arxiv.org/abs/2610.02185)**, focused on the eight same-depth comparisons in **Figure 1b**.

**Status — October 5, 2026:** six comparisons are complete; two AIME comparisons are running. Five completed comparisons improve over their paired baseline; Huginn R32 shows no gain. This is a partial reproduction, not a claim that the paper's scores have been matched. This repository is not affiliated with Apple.

## Results

Scores are percentages; Δ is an absolute percentage-point change. Code tasks use the original benchmark tests, not the extended `Plus` tests. ARC-C and HellaSwag use length-normalized accuracy (`acc_norm`).

<!-- RESULTS:START -->
| Benchmark | Model | Paper baseline → LoopCD | Ours baseline → LoopCD | Δ paper / ours |
|---|---|---:|---:|---:|
| AIME 2024 | Ouro-2.6B Thinking | 61.88 → 73.33 | Running | +11.45 / — |
| AIME 2024 | Ouro-1.4B Thinking | 50.83 → 59.17 | Running | +8.34 / — |
| HumanEval | Huginn R32 | 22.56 → 31.71 | 27.44 → 27.44 | +9.15 / +0.00 |
| HumanEval | Huginn R16 | 20.12 → 28.05 | 22.56 → 26.22 | +7.93 / +3.66 |
| HumanEval | Ouro-2.6B | 75.61 → 79.88 | 73.78 → 80.49 | +4.27 / +6.71 |
| ARC-Challenge | Parcae-370M | 32.59 → 36.95 | 32.34 → 35.32 | +4.36 / +2.99 |
| HellaSwag | Parcae-1.3B | 55.93 → 60.44 | 56.12 → 60.34 | +4.51 / +4.21 |
| MBPP | Looped-Qwen3 | 66.14 → 70.11 | 65.87 → 69.05 | +3.97 / +3.17 |
<!-- RESULTS:END -->

The Looped-Qwen3 row uses a documented **independent reconstruction** of the checkpoint/cache setup. It does not establish the identity of the authors' implementation. The Huginn R32 negative result is retained. Prompt, stopping, initialization and backend differences limit direct numerical comparison; see [protocols and limitations](docs/protocols.md).

[Machine-readable results](results/figure1b.json) · [Statistical evidence](results/evidence) · [Reproduction guide](docs/reproduction.md)

## Method

LoopCD uses an early and a final readout from the same looped model:

```text
fixed logits:     z = z_final + ω (z_final − z_early)
adaptive logits:  ω = cap × (1 − (p_top1 − p_top2))
```

The adaptive margin comes from the full, unmodified final-logit softmax. Guidance is computed in FP32. No training or plausibility mask is added. Baseline and zero guidance preserve native logits. Huginn R32 uses the paper's separate **hidden-state** variant; Huginn R16 uses adaptive **logit-space** guidance.

## Quick start

Python 3.10 and Linux/CUDA are recommended for inference. The validated research environment uses PyTorch 2.9.0 and Transformers 4.54.1. Model weights and datasets are downloaded separately under their original terms.

```bash
git clone https://github.com/tongyao-zhu/reproduction-of-loopcd.git
cd reproduction-of-loopcd
python -m venv .venv
source .venv/bin/activate
pip install -e '.[eval,code,test]'

# No GPU or model download needed: verify all published result arithmetic.
python scripts/verify_results.py

# CPU formula and adapter checks.
pytest -q tests/test_guidance.py tests/test_ouro.py
```

Try the Ouro adapter on one prompt:

```bash
python scripts/prepare_ouro_variant.py --model Ouro-2.6B \
  --cache-dir .cache/huggingface --output models/Ouro-2.6B --download
python examples/generate_ouro.py --model models/Ouro-2.6B \
  --prompt 'Explain why the sum of two even integers is even.'
```

This example demonstrates the adapter; it is **not** a benchmark score. For paired HumanEval generation, isolated scoring, and the status of the remaining research entry points, see [Reproduce](docs/reproduction.md).

## Code map

| Component | Implementation |
|---|---|
| Fixed/adaptive logit guidance | [`guidance.py`](src/loopcd_repro/guidance.py) |
| Ouro native-loop readouts | [`ouro.py`](src/loopcd_repro/ouro.py) |
| Huginn hidden guidance / logit guidance | [`huginn.py`](src/loopcd_repro/huginn.py), [`huginn_logits.py`](src/loopcd_repro/huginn_logits.py) |
| Parcae loop adapter and MC scoring | [`parcae.py`](src/loopcd_repro/parcae.py), [`parcae_mc.py`](src/loopcd_repro/parcae_mc.py), [`parcae370_mc.py`](src/loopcd_repro/parcae370_mc.py) |
| Independent Looped-Qwen3 reconstruction | [`qwen_loop.py`](src/loopcd_repro/qwen_loop.py) |
| vLLM 0.13 Ouro adapter | [`vllm_ouro.py`](src/loopcd_repro/vllm_ouro.py), [`vllm_ouro_plugin.py`](src/loopcd_repro/vllm_ouro_plugin.py) |
| Data preparation, generation, comparison, isolated scoring | [`scripts/`](scripts), [entry-point guide](docs/reproduction.md) |
| Numerical, cache, pairing and scoring checks | [`tests/`](tests) |

The core adapters are unchanged from the research snapshot. This public export omits model weights, generated programs, server logs and private deployment history. Historical launchers retain their provenance checks and require the corresponding local artifacts; they are not a portable one-command benchmark suite. [Source manifest](SOURCE_MANIFEST.json) records original and exported file hashes.

## Citation and license

Please cite the [original paper](https://arxiv.org/abs/2610.02185); bibliographic metadata is in [CITATION.cff](CITATION.cff). Original reproduction code is released under [Apache-2.0](LICENSE). Model weights, datasets and third-party dependencies retain their own licenses; see [third-party notes](THIRD_PARTY.md).
