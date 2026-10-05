#!/usr/bin/env bash
# Launch first milestone: three paired arms, SciQ then PIQA on each GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/path/to/your/workspace/loop_cd_transfer_20260930/venv/bin/python}"
RUN_GROUP="${RUN_GROUP:-$(date -u +%Y%m%dT%H%M%SZ)}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_MODULES_CACHE="$PWD/.cache/modules"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
mkdir -p logs results/runs
mkdir "results/runs/$RUN_GROUP"

# Refuse a busy allocation rather than disturbing an existing experiment.
for gpu in 0 1 2; do
  used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
  if (( used > 1000 )); then
    echo "GPU $gpu is busy ($used MiB); no evaluation was launched." >&2
    exit 1
  fi
done

pids=()
modes=(baseline fixed adaptive)
for gpu in 0 1 2; do
  mode="${modes[$gpu]}"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    for task in sciq piqa; do
      "$PYTHON_BIN" -u scripts/evaluate.py --task "$task" --mode "$mode" \
        --output "results/runs/$RUN_GROUP/${task}_${mode}"
    done
  ) > "logs/${RUN_GROUP}_${mode}.log" 2>&1 &
  pids+=("$!")
done
"$PYTHON_BIN" - "$RUN_GROUP" "${pids[@]}" <<'PY'
import datetime, json, sys
from pathlib import Path
record = {"group": sys.argv[1], "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "worker_pids": dict(zip(("baseline", "fixed", "adaptive"), map(int, sys.argv[2:]))),
          "gpus": {"baseline": 0, "fixed": 1, "adaptive": 2}, "tasks": ["sciq", "piqa"]}
(Path("results/runs") / sys.argv[1] / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record), flush=True)
PY
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if (( failed )); then
  echo "At least one evaluation failed. Inspect individual manifests and logs." >&2
  exit 1
fi
for task in sciq piqa; do
  "$PYTHON_BIN" scripts/compare.py --runs \
    "results/runs/$RUN_GROUP/${task}_baseline" \
    "results/runs/$RUN_GROUP/${task}_fixed" \
    "results/runs/$RUN_GROUP/${task}_adaptive" \
    --output "results/runs/$RUN_GROUP/${task}_comparison.json"
done
