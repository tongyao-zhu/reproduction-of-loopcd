"""Create a pinned private Huginn copy with only cache-property compatibility."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


REPO_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
ORIGINAL_CODE_SHA256 = "a1d447da93c605a6dc11fbc17cbc7185663f4a41ae12335b34b5847edf0f84aa"
MARKER = "class HuginnDynamicCache(DynamicCache):\n"
PATCH = """    # transformers >= 4.54 exposes these names as read-only properties.
    # This native cache owns dictionaries, so retain its original assignments.
    key_cache = None
    value_cache = None
"""


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(source, output):
    """Verify the source before creating anything; never alter the HF cache."""
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    code_source = source / "raven_modeling_minimal.py"
    if sha256(code_source) != ORIGINAL_CODE_SHA256:
        raise ValueError("Pinned Huginn source differs from the audited original")
    code = code_source.read_text()
    if code.count(MARKER) != 1:
        raise ValueError("Unexpected native Huginn cache class layout")
    output.mkdir(parents=True)
    files = {}
    for path in sorted(source.iterdir()):
        if not path.is_file():
            continue
        destination = output / path.name
        if path.suffix in (".safetensors", ".bin", ".pt"):
            destination.symlink_to(path.resolve())
            files[path.name] = {"weight_blob": path.resolve().name, "bytes": path.stat().st_size}
        else:
            shutil.copyfile(path, destination)
            files[path.name] = {"sha256_original": sha256(path)}
    private_code = output / "raven_modeling_minimal.py"
    private_code.write_text(code.replace(MARKER, MARKER + PATCH))
    record = {
        "repo_id": REPO_ID, "revision": REVISION, "files": files,
        "model_code_sha256": sha256(private_code),
        "compatibility_patch": "shadow read-only key_cache/value_cache properties for transformers 4.54+",
        "model_semantics": "native initialization and recurrence unchanged; private code; read-only weight reuse",
    }
    (output / "model_provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("models/huginn-0125"))
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    from huggingface_hub import snapshot_download
    source = snapshot_download(REPO_ID, revision=REVISION, cache_dir=args.cache_dir,
                               local_files_only=not args.download)
    record = prepare(source, args.output)
    print(json.dumps({"model_path": str(args.output.resolve()), "revision": record["revision"]}))


if __name__ == "__main__":
    main()
