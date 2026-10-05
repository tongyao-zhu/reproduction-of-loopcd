"""Model loading and run provenance, with no shared-cache modifications."""
import hashlib
import importlib.metadata
import inspect
import json
import platform
import subprocess
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_ouro(model_path, device="cuda:0"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = Path(model_path).resolve()
    if not (path / "model_provenance.json").is_file():
        raise ValueError("Run scripts/prepare_model.py first: a pinned local model is required")
    model = AutoModelForCausalLM.from_pretrained(
        str(path), trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map={"": device}, local_files_only=True, attn_implementation="sdpa",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    return model, tokenizer


def provenance(model_path, model=None):
    import torch
    root = Path(__file__).resolve().parents[2]
    result = {
        "python": platform.python_version(),
        "packages": {key: importlib.metadata.version(key) for key in
                     ("torch", "transformers", "accelerate", "datasets", "lm_eval")},
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "model": json.loads((Path(model_path) / "model_provenance.json").read_text()),
        "source_sha256": {str(p.relative_to(root)): sha256(p)
                          for folder in ("src", "scripts", "configs")
                          for p in sorted((root / folder).rglob("*"))
                          if p.is_file() and p.suffix in (".py", ".json", ".yaml")},
        "arxiv": "2610.02185v1",
        "precision": "BF16 model; FP32 guidance/log_softmax",
        "attention": "sdpa",
    }
    if model is not None:
        result["loaded_model_code_sha256"] = sha256(inspect.getfile(type(model)))
    try:
        result["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        result["git_commit"] = None
    return result
