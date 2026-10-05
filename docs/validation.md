# Public snapshot validation — 2026-10-05

- Published result verification: PASS (six complete comparisons, two pending; evidence hashes, integer counts, paired deltas and README agree).
- Python syntax: 174 source, research-script, test and example files compiled before publication; no inference was run for this check.
- README/document local links and all 179 exported source/test hashes: PASS. All numerical modules under `src/` are byte-identical to the research snapshot.
- CPU adapter suite: **72 passed, 32 skipped, 16 subtests passed** (22.13 seconds). Command: `pytest -q -p no:cacheprovider tests/test_guidance.py tests/test_ouro.py tests/test_huginn.py tests/test_huginn_logits.py tests/test_parcae.py tests/test_qwen_loop.py`. The suite conditionally skipped 32 checks; these are not counted as verified. GPU visibility was disabled.
- Ouro demonstration: command-line help checked; no new full-checkpoint inference test was run during publication. The underlying adapter is unchanged from the tested research code.
- No full benchmark rerun or raw-output rescore was performed as part of publication. Historical integration workflows require their source-bound artifacts, described in the reproduction guide.
