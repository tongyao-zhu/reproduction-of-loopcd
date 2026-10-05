"""Pin Ouro model source; copy code, link weights, patch only the private copy."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

REVISION = "574fa66cb8bf5abdc979642d01cf2b79b16bfab1"
ORIGINAL_CODE_SHA256 = "c5c68fbb368ce2909c257ae2afc50719be8c91539333d3295e19312c4316f413"
CACHE_COMPAT = '''
    # Compatibility with transformers 4.54: native cache uses flat lists.
    key_cache = None
    value_cache = None

    def get_mask_sizes(self, cache_position, layer_idx):
        new_tokens = cache_position.shape[-1] if cache_position is not None else 1
        return self._seen_tokens + new_tokens, 0

'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("models/Ouro-1.4B"))
    parser.add_argument("--download", action="store_true", help="Allow fetching the pinned public model")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}; choose a new output")
    from huggingface_hub import snapshot_download
    source = Path(snapshot_download(
        "ByteDance/Ouro-1.4B", revision=REVISION, cache_dir=args.cache_dir,
        local_files_only=not args.download,
    ))
    source_code = source / "modeling_ouro.py"
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if digest(source_code) != ORIGINAL_CODE_SHA256:
        raise ValueError("Pinned Ouro source differs from audited original; inspect before patching")
    args.output.mkdir(parents=True)
    files = {}
    for path in source.iterdir():
        if not path.is_file():
            continue
        dest = args.output / path.name
        if path.suffix in (".safetensors", ".bin", ".pt"):
            dest.symlink_to(path.resolve())
            files[path.name] = {"weight_blob": path.resolve().name, "bytes": path.stat().st_size}
        else:
            shutil.copyfile(path, dest)
            files[path.name] = {"sha256_original": digest(path)}
    code_path = args.output / "modeling_ouro.py"
    code = code_path.read_text()
    marker = '    def __init__(self, max_cache_size: Optional[int] = None):'
    if code.count(marker) != 1:
        raise ValueError("Unexpected cache source layout")
    code_path.write_text(code.replace(marker, CACHE_COMPAT + marker))
    record = {
        "repo_id": "ByteDance/Ouro-1.4B", "revision": REVISION,
        "files": files, "model_code_sha256": digest(code_path),
        "compatibility_patch": "flat cache properties and mask-size interface for transformers 4.54.1",
        "model_semantics": "unchanged; code is private copy; weight blobs read-only",
    }
    (args.output / "model_provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"model_path": str(args.output.resolve()), "revision": REVISION}))


if __name__ == "__main__":
    main()
