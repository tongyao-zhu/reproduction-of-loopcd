"""Prepare pinned Parcae-370M files using only the Python standard library.

Offline and CPU-only: this script never imports model code, deserializes a
checkpoint, downloads files, installs packages, or changes the shared cache.
Only a fresh output directory is accepted. An interrupted copy is cleaned up
and leaves preparation_failure.json in the reserved output for diagnosis;
that directory must not be reused as a successful prepared model.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys


MODEL_REPO = "SandyResearch/parcae-370m"
MODEL_REVISION = "439284464ee4999bd1f762da7d044613a4828efe"
TOKENIZER_REPO = "SandyResearch/parcae-tokenizer"
TOKENIZER_REVISION = "6247b5d0592876b73660f34c2dd16c9db9e9045c"
SOURCE_REPO = "https://github.com/sandyresearch/parcae"
SOURCE_REVISION = "69284c13746e849104f738d6d1a347b1f457df76"
SOURCE_TREE_SHA1 = "1e284593583eaf4403dad3bd5054ea81eb0cb7ac"
CHUNK_BYTES = 8 * 1024 * 1024
FILES = {
    "pytorch_model.bin": {"repo_id": MODEL_REPO, "revision": MODEL_REVISION,
                          "bytes": 1553099635,
                          "sha256": "603d9da4a1c1a112c8b6a98bc1e9aac288990ba0d7f5b432aaad9c53940bfcb2"},
    "config.json": {"repo_id": MODEL_REPO, "revision": MODEL_REVISION,
                    "bytes": 1745,
                    "sha256": "0ed6862c495bfb670fc72eba955c555c472a5dcdf33bed6852608135ead6666e"},
    "tokenizer.json": {"repo_id": TOKENIZER_REPO, "revision": TOKENIZER_REVISION,
                       "bytes": 2326980,
                       "sha256": "e0021e26057088d68047dbe6e77e2ba1c9fe9ae45bae3df9d67d48c82405ea77"},
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_identity(path):
    info = Path(path).stat()
    return {"device": info.st_dev, "inode": info.st_ino, "bytes": info.st_size,
            "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns}


def git_tree_sha1(files):
    """Reconstruct a Git tree from verified regular blobs, without a checkout."""
    tree = {}
    for name, record in files.items():
        parts = PurePosixPath(name).parts
        if not parts or PurePosixPath(name).is_absolute() or ".." in parts or ".git" in parts:
            raise ValueError(f"Invalid source path: {name}")
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError("Source tree file/directory collision")
        if parts[-1] in node or record["git_mode"] not in ("100644", "100755"):
            raise ValueError("Duplicate or unsupported source tree entry")
        node[parts[-1]] = (record["git_mode"], record["git_blob_sha1"])

    def encode(node):
        parts = []
        for name, child in sorted(node.items(), key=lambda item: os.fsencode(item[0]) + (b"/" if isinstance(item[1], dict) else b"")):
            mode, object_id = ("40000", encode(child)) if isinstance(child, dict) else child
            parts.append(mode.encode("ascii") + b" " + os.fsencode(name) + b"\0" + bytes.fromhex(object_id))
        payload = b"".join(parts)
        return hashlib.sha1(f"tree {len(payload)}\0".encode("ascii") + payload).hexdigest()
    return encode(tree)


def stream_fingerprint(path, *, chunk_bytes=CHUNK_BYTES, git_blob=False):
    """Read bounded chunks; also verify source stability while hashing."""
    path = Path(path)
    if type(chunk_bytes) is not int or chunk_bytes < 1:
        raise ValueError("chunk_bytes must be a positive integer")
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"Expected a regular file: {path}")
    sha256 = hashlib.sha256()
    blob = hashlib.sha1(f"blob {before.st_size}\0".encode("ascii")) if git_blob else None
    count = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            count += len(chunk)
            sha256.update(chunk)
            if blob is not None:
                blob.update(chunk)
    after = path.stat()
    identity = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    if count != before.st_size or identity(before) != identity(after):
        raise ValueError(f"Source changed while being hashed: {path}")
    result = {"bytes": count, "sha256": sha256.hexdigest()}
    if blob is not None:
        result["git_blob_sha1"] = blob.hexdigest()
    return result


def run_git(source_root, *args):
    # Status checks must not refresh/write the index or invoke fsmonitor hooks.
    result = subprocess.run(
        ["git", "-c", "core.fsmonitor=false", "-C", str(source_root), *args],
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"}, capture_output=True, check=False,
    )
    if result.returncode:
        raise ValueError(f"Git inspection failed ({' '.join(args)}): {result.stderr.decode(errors='replace').strip()}")
    return result.stdout


def check_source_identity(source_root):
    root = Path(source_root).resolve(strict=True)
    top = Path(os.fsdecode(run_git(root, "rev-parse", "--show-toplevel").strip())).resolve()
    if root != top:
        raise ValueError("--source-root must be the exact official checkout root")
    head = run_git(root, "rev-parse", "HEAD").decode("ascii").strip()
    if head != SOURCE_REVISION:
        raise ValueError(f"Official source must be exactly {SOURCE_REVISION}; found {head}")
    if run_git(root, "rev-parse", "HEAD^{tree}").decode("ascii").strip() != SOURCE_TREE_SHA1:
        raise ValueError("Official source Git tree differs from the registered commit tree")
    dirty = run_git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise ValueError("Official source checkout has modified, staged, or untracked files")
    return root


def inspect_source(source_root):
    root = check_source_identity(source_root)
    entries = run_git(root, "ls-tree", "-r", "-z", "HEAD").split(b"\0")
    files = {}
    for entry in entries:
        if not entry:
            continue
        header, encoded_name = entry.split(b"\t", 1)
        mode, kind, object_id = header.decode("ascii").split()
        name = os.fsdecode(encoded_name)
        parts = PurePosixPath(name).parts
        if not parts or PurePosixPath(name).is_absolute() or ".." in parts or ".git" in parts:
            raise ValueError(f"Unexpected source path: {name}")
        if mode not in ("100644", "100755") or kind != "blob":
            raise ValueError(f"Source symlinks/submodules are not supported by this preparation: {name}")
        path = root / name
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
            raise ValueError(f"Source file escapes the pinned checkout: {name}")
        actual = stream_fingerprint(path, git_blob=True)
        if actual["git_blob_sha1"] != object_id:
            raise ValueError(f"Source content differs from the pinned Git tree: {name}")
        files[name] = {**actual, "git_mode": mode}
    if not files or not any(name.startswith("parcae_lm/") for name in files):
        raise ValueError("Official checkout is missing the parcae_lm package")
    if git_tree_sha1(files) != SOURCE_TREE_SHA1:
        raise ValueError("Verified source files do not reconstruct the pinned Git tree")
    check_source_identity(root)
    return root, files


def locate_cached_file(cache_dir, filename, spec):
    snapshot = cache_dir / ("models--" + spec["repo_id"].replace("/", "--")) / "snapshots" / spec["revision"]
    path = snapshot / filename
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(cache_dir) or not resolved.is_file():
        raise ValueError(f"Pinned cache entry must resolve to a regular file inside --cache-dir: {path}")
    if resolved.stat().st_size != spec["bytes"]:
        raise ValueError(f"Pinned source size mismatch: {filename}")
    return path, resolved


def verify_cached_file(cache_dir, filename, spec):
    logical, resolved = locate_cached_file(cache_dir, filename, spec)
    identity = file_identity(resolved)
    actual = stream_fingerprint(resolved)
    if actual != {"bytes": spec["bytes"], "sha256": spec["sha256"]}:
        raise ValueError(f"Pinned source SHA256 mismatch: {filename}; observed {actual['sha256']}")
    if file_identity(resolved) != identity:
        raise ValueError(f"Cache entry changed while being verified: {filename}")
    return {**spec, **actual, "snapshot_path": str(logical), "resolved_path": str(resolved),
            "source_file_identity": identity,
            "url": f"https://huggingface.co/{spec['repo_id']}/resolve/{spec['revision']}/{filename}"}


def verify_prepared(output):
    """Rehash all prepared bytes and the pinned Git tree; never load a model.

    This independently authenticates the source file set against a fixed tree
    hash, so editing both a private source file and its manifest cannot pass.
    The original checkout is not required after successful preparation.
    """
    root = Path(output).resolve(strict=True)
    manifest = json.loads((root / "model_provenance.json").read_text())
    if (manifest.get("schema_version") != 1 or manifest.get("status") != "PASS"
            or manifest.get("kind") != "parcae_model_preparation"
            or manifest.get("repo_id") != MODEL_REPO or manifest.get("revision") != MODEL_REVISION
            or manifest.get("tokenizer") != {"repo_id": TOKENIZER_REPO, "revision": TOKENIZER_REVISION}):
        raise ValueError("Prepared model manifest is not the pinned complete Parcae-370M preparation")
    if {path.name for path in root.iterdir()} != set(FILES) | {"source", "model_provenance.json"}:
        raise ValueError("Unexpected or missing files in prepared model directory")
    if set(manifest.get("files", {})) != set(FILES):
        raise ValueError("Prepared manifest file set changed")
    for name, spec in FILES.items():
        record = manifest["files"][name]
        if any(record.get(key) != value for key, value in spec.items()):
            raise ValueError(f"Prepared manifest file pin changed: {name}")
        path = root / name
        if name == "pytorch_model.bin":
            if not path.is_symlink() or path.resolve(strict=True) != Path(record["resolved_path"]):
                raise ValueError("Checkpoint is not the recorded read-only-use blob link")
        elif path.is_symlink():
            raise ValueError("Prepared metadata must be a private regular-file copy")
        if stream_fingerprint(path) != {key: spec[key] for key in ("bytes", "sha256")}:
            raise ValueError(f"Prepared bytes changed: {name}")
    source = manifest.get("source", {})
    if (source.get("git_commit") != SOURCE_REVISION or source.get("git_tree_sha1") != SOURCE_TREE_SHA1
            or source.get("private_relative_root") != "source" or source.get("repo_url") != SOURCE_REPO):
        raise ValueError("Prepared source identity changed")
    code = root / "source"
    if code.is_symlink() or not code.is_dir():
        raise ValueError("Prepared source must be a private directory")
    files = {}
    for path in code.rglob("*"):
        if path.is_symlink():
            raise ValueError("Prepared source contains an unexpected symlink")
        if path.is_dir():
            continue
        name = path.relative_to(code).as_posix()
        mode = "100755" if path.stat().st_mode & 0o111 else "100644"
        files[name] = {**stream_fingerprint(path, git_blob=True), "git_mode": mode}
    if files != source.get("files") or git_tree_sha1(files) != SOURCE_TREE_SHA1:
        raise ValueError("Prepared source bytes, modes, or file collection differ from the fixed Git tree")
    return manifest


def prepare(cache_dir, source_root, output):
    """Create a private, fully byte-verified copy without executing source code."""
    cache_dir = Path(cache_dir).expanduser().resolve(strict=True)
    source_root = Path(source_root).expanduser().resolve(strict=True)
    output = Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite any existing output: {output}")
    destination = output.resolve()
    if destination.is_relative_to(cache_dir) or destination.is_relative_to(source_root):
        raise ValueError("Output must be outside both the shared cache and official source checkout")
    # Cheap identity/size failures occur before hashing the 1.55 GB checkpoint.
    for filename, spec in FILES.items():
        locate_cached_file(cache_dir, filename, spec)
    root, source_files = inspect_source(source_root)
    verified = {}
    for filename in ("config.json", "tokenizer.json", "pytorch_model.bin"):
        verified[filename] = verify_cached_file(cache_dir, filename, FILES[filename])
    output.mkdir(parents=True, exist_ok=False)
    phase = "copy_nonweight_files"
    try:
        for filename in ("config.json", "tokenizer.json"):
            shutil.copyfile(verified[filename]["resolved_path"], output / filename)
            if stream_fingerprint(output / filename) != {key: verified[filename][key] for key in ("bytes", "sha256")}:
                raise ValueError(f"Copied file failed verification: {filename}")
        phase = "copy_official_source"
        for name, expected in source_files.items():
            destination_file = output / "source" / name
            destination_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, destination_file)
            destination_file.chmod(0o755 if expected["git_mode"] == "100755" else 0o644)
            if stream_fingerprint(destination_file) != {key: expected[key] for key in ("bytes", "sha256")}:
                raise ValueError(f"Copied source failed verification: {name}")
        phase = "link_verified_checkpoint"
        weight = Path(verified["pytorch_model.bin"]["resolved_path"])
        (output / "pytorch_model.bin").symlink_to(weight)
        # The snapshot itself must still resolve to the file actually verified.
        for filename, evidence in verified.items():
            current = Path(evidence["snapshot_path"]).resolve(strict=True)
            if current != Path(evidence["resolved_path"]) or file_identity(current) != evidence["source_file_identity"]:
                raise ValueError(f"Cache identity changed during preparation: {filename}")
        check_source_identity(root)
        phase = "write_manifest"
        manifest = {
            "schema_version": 1, "status": "PASS", "kind": "parcae_model_preparation",
            "created_utc": utc_now(), "repo_id": MODEL_REPO, "revision": MODEL_REVISION,
            "tokenizer": {"repo_id": TOKENIZER_REPO, "revision": TOKENIZER_REVISION},
            "files": {name: {**record, "storage": "symlink" if name == "pytorch_model.bin" else "copy"}
                      for name, record in verified.items()},
            "source": {"repo_url": SOURCE_REPO, "git_commit": SOURCE_REVISION,
                       "git_tree_sha1": SOURCE_TREE_SHA1,
                       "original_root": str(root), "private_relative_root": "source",
                       "working_tree_clean": True, "tracked_git_blobs_verified": True,
                       "files": source_files},
            "preparation_script_sha256": stream_fingerprint(Path(__file__))["sha256"],
            "validation": {"all_checkpoint_bytes_hashed": True, "config_and_tokenizer_hashed": True,
                           "private_copies_rehashed": True, "source_imported": False,
                           "checkpoint_deserialized": False, "model_loaded": False, "gpu_used": False},
            "weight_access": "Shared blob linked for read-only consumption; symlinks do not enforce filesystem write protection. Never modify or chmod the shared target.",
            "downstream_load_requirements": {"torch_load_weights_only": True,
                                             "check_all_missing_and_unexpected_keys": True,
                                             "python_minimum": "3.11"},
            "omitted_cache_files": "Only the three explicitly pinned files are prepared; token_bytes.pt and other cache entries are not loaded or copied.",
            "scope": "Byte/source integrity preparation only; loading compatibility, tokenizer semantics, numerical correctness and GPU readiness remain unverified.",
        }
        (output / "model_provenance.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return manifest
    except BaseException as exc:
        # These names were reserved by this fresh-only invocation. Keep a small
        # diagnostic marker but remove incomplete private code/weights links.
        cleanup_errors = []
        for name in ("source", "config.json", "tokenizer.json", "pytorch_model.bin", "model_provenance.json"):
            path = output / name
            try:
                if path.is_symlink() or path.is_file():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)
            except OSError as cleanup_error:
                cleanup_errors.append(f"{name}: {cleanup_error!r}")
        failure = {"status": "FAIL", "phase": phase, "error": repr(exc), "at_utc": utc_now(),
                   "repo_id": MODEL_REPO, "revision": MODEL_REVISION,
                   "source_git_commit": SOURCE_REVISION, "cleanup_errors": cleanup_errors,
                   "instruction": "Incomplete preparation. Preserve this diagnostic; choose a fresh output directory for retry."}
        try:
            (output / "preparation_failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        except OSError as marker_error:
            print(f"Could not preserve preparation failure marker: {marker_error!r}", file=sys.stderr)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        record = prepare(args.cache_dir, args.source_root, args.output)
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "output": str(args.output), "error": str(exc)}), file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps({"status": record["status"], "output": str(args.output.resolve()),
                      "repo_id": record["repo_id"], "revision": record["revision"],
                      "source_git_commit": record["source"]["git_commit"],
                      "weight_sha256": record["files"]["pytorch_model.bin"]["sha256"]}))


if __name__ == "__main__":
    main()
