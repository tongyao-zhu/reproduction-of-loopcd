"""Prepare audited Ouro snapshots without changing shared files or loading a model.

The default path is offline and reads the cache directly. The explicit
--download flag may populate that cache; all compatibility edits stay private.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil


COMMON_HASHES = {
    "merges.txt": "0b54e8aa4e53d5383e2e4bc635a56b43f9647f7b13832d5d9ecd8f82dac4f510",
    "tokenizer.json": "fcb808fe5e7642f5299be28aea07fc7f6d4f4364c3ac5e408e15a772cbc8fa8d",
    "vocab.json": "7b9de3f47796abf8d00ab96be299fea0dc9afdf1827f34e7e0b9fb44593efe5c",
}
BASE_CODE = "c5c68fbb368ce2909c257ae2afc50719be8c91539333d3295e19312c4316f413"
BASE_CONFIG_CODE = "950443e32929047aa08d02abad2e1888bc1914b3db988d3d675f70787f65dafb"
BASE_TOKENIZER = "7e010da95d71b0fa1aa809552922c74cc1dd98d628f21ba39ca6c35aa18d5d91"
BASE_SPECIAL = "55087bb8409060d9cb0f80495e34f4e0d8a84f68a799a6b0cab3132be0aae319"
THINKING_SPECIAL = "688ac94df221b18a4d2d4579dfa3c4f0fd5ca3e394da46cd0a317fb5d57d4685"


def specification(revision, layers, code, config_code, config, tokenizer, special, native_mask):
    return {
        "revision": revision, "layers": layers, "native_mask_sizes": native_mask,
        "weights_bytes": 2869336434 if layers == 24 else 5336011242,
        "audited_files": {**COMMON_HASHES, "modeling_ouro.py": code,
                          "configuration_ouro.py": config_code, "config.json": config,
                          "tokenizer_config.json": tokenizer, "special_tokens_map.json": special},
    }


VARIANTS = {
    "Ouro-1.4B": specification(
        "574fa66cb8bf5abdc979642d01cf2b79b16bfab1", 24, BASE_CODE, BASE_CONFIG_CODE,
        "ce9cc13da41591b8b4deca053d7dfee06424c0228628ee862ea86d725bc163f3",
        BASE_TOKENIZER, BASE_SPECIAL, False),
    "Ouro-2.6B": specification(
        "1ed04250da1a9936042725d302e81c8fa2ab5abd", 48, BASE_CODE, BASE_CONFIG_CODE,
        "23b5c6942a9f0ca8619fb15acfaca93e02c169cdc4e03dbceb4d8a2df145697a",
        BASE_TOKENIZER, BASE_SPECIAL, False),
    "Ouro-1.4B-Thinking": specification(
        "3aaa2224253a92ca45cf2e3d427c360e1ef9c93d", 24,
        "cb8c980b016ae8ae35b6c1193633b9b4496e89993e3a0e8c7853324f01904b30", BASE_CONFIG_CODE,
        "44e42ec2ed97f49f20f87985688b6e1be9dd61bbfe9f8a349daac192126e2c55",
        "e41eafc6bf7269d6f90f2531124949073a57965f5f0765238a744c1c528e2448",
        THINKING_SPECIAL, True),
    "Ouro-2.6B-Thinking": specification(
        "f1edd81e7ac41355db670500ceaf204e0f73af68", 48,
        "263c097a2ed6870ce549a65ef6d6bb9e8e167423f87d2c494e0988b9a4f63f4b",
        "ba60253335cecc5bd9cb3e5f8cfbd214aa7e865fc97a98208976fb2517811de6",
        "ad4b51dffae60bebeeac5e1d5055e78d01fc01b1a29cd89cd2e50edb848f645a",
        "0d304de54d3f8e0c0573d678c0f1d80cc26e9d23dcb328405f0ec99c4293d04f",
        THINKING_SPECIAL, True),
}

MARKER = "    def __init__(self, max_cache_size: Optional[int] = None):"
PROPERTIES = """
    # Compatibility with transformers 4.54: native cache uses flat lists.
    key_cache = None
    value_cache = None

"""
MASK_SIZES = """    def get_mask_sizes(self, cache_position, layer_idx):
        new_tokens = cache_position.shape[-1] if cache_position is not None else 1
        return self._seen_tokens + new_tokens, 0

"""


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_snapshot(model, cache_dir, download=False):
    spec = VARIANTS[model]
    cache_dir = Path(cache_dir).expanduser().resolve()
    if download:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download("ByteDance/" + model, revision=spec["revision"],
                                      cache_dir=str(cache_dir)))
    path = cache_dir / ("models--ByteDance--" + model) / "snapshots" / spec["revision"]
    if not path.is_dir():
        raise FileNotFoundError(f"Pinned snapshot not cached: {path}; downloading requires --download")
    return path


def prepare(source, output, model):
    """Validate identity and completeness before creating the private directory."""
    spec = VARIANTS[model]
    source, output = Path(source).absolute(), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if (source.name != spec["revision"] or source.parent.name != "snapshots"
            or source.parent.parent.name != "models--ByteDance--" + model):
        raise ValueError("Source must be this model's exact pinned Hub snapshot")
    if output.resolve().is_relative_to(source.parents[2].resolve()):
        raise ValueError("Output must be outside the shared Hub cache")
    for name, expected in spec["audited_files"].items():
        path = source / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Pinned {model} source missing or changed: {name}")
    weights = source / "model.safetensors"
    if not weights.is_file() or weights.stat().st_size != spec["weights_bytes"]:
        raise ValueError("Pinned single-file checkpoint is missing or incomplete")
    if any(p.suffix in (".safetensors", ".bin", ".pt") and p != weights for p in source.iterdir()):
        raise ValueError("Unexpected additional checkpoint files in pinned snapshot")
    config = json.loads((source / "config.json").read_text())
    expected_config = {"architectures": ["OuroForCausalLM"], "model_type": "ouro",
                       "num_hidden_layers": spec["layers"], "hidden_size": 2048,
                       "total_ut_steps": 4, "early_exit_threshold": 1.0}
    for key, value in expected_config.items():
        if config.get(key) != value:
            raise ValueError(f"Unexpected model configuration: {key}")
    original = (source / "modeling_ouro.py").read_text()
    if original.count(MARKER) != 1:
        raise ValueError("Unexpected native cache constructor")
    if original.count("    def get_mask_sizes(") != int(spec["native_mask_sizes"]):
        raise ValueError("Unexpected native cache mask interface")
    addition = PROPERTIES + ("" if spec["native_mask_sizes"] else MASK_SIZES)
    patched = original.replace(MARKER, addition + MARKER)
    compile(patched, "modeling_ouro.py", "exec")
    # A dangling source entry is a partial snapshot, even if it is ancillary.
    if any(p.is_symlink() and not p.exists() for p in source.iterdir()):
        raise ValueError("Pinned snapshot contains dangling entries")
    output.mkdir(parents=True)
    files = {}
    for path in sorted(source.iterdir()):
        if not path.is_file():
            continue
        destination = output / path.name
        if path == weights:
            destination.symlink_to(path.resolve())
            files[path.name] = {"weight_blob": path.resolve().name, "bytes": path.stat().st_size}
        else:
            shutil.copyfile(path, destination)
            files[path.name] = {"sha256_original": sha256(path)}
    (output / "modeling_ouro.py").write_text(patched)
    patch = "".join(difflib.unified_diff(original.splitlines(True), patched.splitlines(True),
                                        fromfile="original/modeling_ouro.py", tofile="private/modeling_ouro.py"))
    record = {
        "repo_id": "ByteDance/" + model, "revision": spec["revision"], "files": files,
        "model_code_sha256": sha256(output / "modeling_ouro.py"),
        "compatibility_patch": ("flat cache property compatibility for transformers 4.54.1; "
                                + ("preserve native get_mask_sizes" if spec["native_mask_sizes"]
                                   else "add mask-size interface matching the validated base adapter")),
        "compatibility_patch_diff": patch,
        "compatibility_patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
        "model_semantics": "recurrence/readout unchanged; private code; read-only weight reuse",
        "validation": {"pinned_source_sha256": True, "checkpoint_size_verified": True,
                       "weight_content_rehashed": False, "real_model_smoke_required": True},
    }
    (output / "model_provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=VARIANTS, default="Ouro-2.6B")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    record = prepare(resolve_snapshot(args.model, args.cache_dir, args.download), args.output, args.model)
    print(json.dumps({"model_path": str(args.output.resolve()), "repo_id": record["repo_id"],
                      "revision": record["revision"], "model_code_sha256": record["model_code_sha256"]}))


if __name__ == "__main__":
    main()
