# Reproduce and inspect

Start with the result check below; it needs only Python. To run inference, install the dependencies and prepare the model and data. Huginn R32 has a generation example here. The other benchmark runners need additional artifacts from the original runs, listed in section 4.

## 1. Verify published statistics

```bash
python scripts/verify_results.py
```

This needs only Python. It verifies the evidence-file hashes, all eight completed comparisons, including AIME’s 30 × 16 sample counts and deltas, paired wins/losses/ties. It also checks the README table against the JSON. It does not reproduce model inference.

## 2. Install and check the adapters

Use Python 3.10 on Linux for the recorded environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[eval,code,test]'
pytest -q tests/test_guidance.py tests/test_ouro.py tests/test_huginn.py \
  tests/test_huginn_logits.py tests/test_parcae.py tests/test_qwen_loop.py
```

Some checks depend on native model-code fixtures or fixture capabilities and skip when those prerequisites are absent. A CPU test pass is not a real-checkpoint GPU certificate. The full tests directory also contains historical integration tests requiring unpublished local run artifacts.

For an adapter-only example, run the preparation and `examples/generate_ouro.py` commands in the README. Preparation downloads a pinned public revision, creates a local compatibility copy, and preserves the original weight blobs. `trust_remote_code` is used for these model implementations.

## 3. Paired Huginn R32 HumanEval generation

Prepare the pinned Huginn checkpoint:

```bash
python scripts/prepare_huginn.py --download --output models/huginn-0125
```

Acquire HumanEvalPlus-v0.1.10 from the upstream [EvalPlus data release](https://github.com/evalplus/evalplus). The uncompressed JSONL must have SHA256 `42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f`; put it at `data/HumanEvalPlus-v0.1.10.jsonl`. The generator rejects a different file.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/generate_humaneval.py \
  --model models/huginn-0125 --data data/HumanEvalPlus-v0.1.10.jsonl \
  --output results/runs/huginn-r32 --arms baseline32 hidden32 \
  --max-new-tokens 2048 --seed 42
```

Use a fresh output directory. `--limit` is only for debugging and is not a full benchmark. The generator saves manifests, prompts, seeds, token IDs and sanitized solutions; it does not execute those solutions. Code scoring is a separate step below.

## 4. Other Figure 1b entry points

| Comparison | Generation / evaluation | Validation / comparison |
|---|---|---|
| Huginn R16 | `generate_humaneval_logits.py` | `gpu_smoke_huginn_logits.py`, `humaneval_logits_checks.py`, `score_humaneval_logits.py` |
| Ouro HumanEval | `generate_ouro_humaneval.py` | `gpu_smoke_ouro_humaneval.py`, `ouro_humaneval_checks.py`, `score_ouro_humaneval.py` |
| Parcae HellaSwag | `evaluate_parcae.py` | `audit_parcae_prompts.py`, `gpu_smoke_parcae.py`, `compare_parcae_figure1b.py` |
| Parcae ARC-C | `evaluate_parcae370.py` | `audit_parcae370_prompts.py`, `gpu_smoke_parcae370.py`, `compare_parcae370.py` |
| Qwen MBPP | `generate_qwen_mbpp.py` | `gpu_smoke_qwen_loop_v3.py`, `qwen_stream_oracle.py`, `score_qwen_mbpp.py` |
| AIME vLLM | `run_vllm_aime.py`, `run_vllm_aime_shard.py` | `gate_vllm_reference.py`, `probe_vllm_aime_budget.py`, `score_sharded_aime.py` |

These runners check their inputs against the original source snapshots and validation records. Some require `releases/` directories, prompt audits, GPU checks and previous batch manifests that are not included here. Their historical commit IDs refer to the research archive. Machine paths have been replaced by `/path/to/your/workspace`.

For a fresh run, prepare the model and inputs, freeze the source, run the corresponding GPU checks, then check generation and scoring on two tasks before starting the full benchmark. These seven workflows still need that setup; we have not rerun them end to end from a fresh clone of this repository.

## 5. Isolated code scoring

The original HumanEval/MBPP scorers use Linux chroot, a dedicated unused UID/GID, seccomp, resource limits and an offline runtime. Preparation requires root inside an appropriate dedicated evaluation environment. Do not run generated programs directly on your development machine.

For HumanEval, the sequence is:

```bash
# In a dedicated Linux evaluation environment with the installed Python packages
# and libseccomp.so.2. Preparation requires root; no shared environment is changed.
python scripts/prepare_eval_sandbox.py --dataset data/HumanEvalPlus-v0.1.10.jsonl
python scripts/run_eval_sandbox.py --self-test \
  --output results/scores/sandbox-safety.json
python scripts/run_eval_sandbox.py --canonical-all \
  --output results/scores/sandbox-canonical164.json
python scripts/run_eval_sandbox.py \
  --samples results/runs/huginn-r32/baseline32/samples.jsonl \
  --output results/scores/huginn-r32-baseline.json
python scripts/run_eval_sandbox.py \
  --samples results/runs/huginn-r32/hidden32/samples.jsonl \
  --output results/scores/huginn-r32-hidden.json
```

The runner fails closed if isolation or canonical validation is missing. See `prepare_mbpp_sandbox.py` / `run_mbpp_sandbox.py` for the corresponding 378-task MBPP runner. These scripts are included as research tooling, not as a security guarantee for a general multi-tenant service.

## Source and result provenance

`SOURCE_MANIFEST.json` pins the clean public export to a local research snapshot and records per-file hashes. Numerical adapters under `src/` are byte-identical to that snapshot. Public result files are explicitly marked statistical extracts and retain each original file's SHA256. The full raw program/text archive, weights, local caches and process logs are not redistributed.

## Rebuild the comparison figure

After verifying the result extracts, install Matplotlib and run `python scripts/plot_figure1b.py`. It reads the same results JSON and writes PNG, SVG and PDF files under `figures/`. No inference is needed.
