"""CPU/offline tests using tiny bytes and an isolated temporary Git checkout."""
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_parcae.py"
SPEC = importlib.util.spec_from_file_location("prepare_parcae", SCRIPT)
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache, self.source, self.output = (self.root / name for name in ("hub", "official", "private"))
        self.cache.mkdir()
        self.source.mkdir()
        (self.source / "parcae_lm").mkdir()
        (self.source / "parcae_lm/__init__.py").write_text("raise AssertionError('Model source must never be imported')\n")
        (self.source / "README.md").write_text("fixture source\n")
        (self.source / "a").mkdir()
        (self.source / "a/b.txt").write_text("directory sort fixture")
        (self.source / "a.c").write_text("filename sort fixture")
        (self.source / "run.sh").write_text("exit 99\n")
        (self.source / "run.sh").chmod(0o755)
        self.git("init", "-q")
        self.git("add", ".")
        self.git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture")
        self.commit = self.git("rev-parse", "HEAD").strip()
        self.tree = self.git("rev-parse", "HEAD^{tree}").strip()
        files = deepcopy(prepare.FILES)
        self.originals = {}
        for name, data in {"pytorch_model.bin": b"arbitrary non-pickle bytes, never deserialized",
                           "config.json": b'{"block_size":2048}\n',
                           "tokenizer.json": b'{"fixture":"no tokenizer code executed"}\n'}.items():
            spec = files[name]
            spec.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            repository = self.cache / ("models--" + spec["repo_id"].replace("/", "--"))
            blob = repository / "blobs" / spec["sha256"]
            blob.parent.mkdir(parents=True, exist_ok=True)
            blob.write_bytes(data)
            snapshot = repository / "snapshots" / spec["revision"]
            snapshot.mkdir(parents=True, exist_ok=True)
            (snapshot / name).symlink_to(blob)
            self.originals[name] = blob
        for attribute, value in (("SOURCE_REVISION", self.commit), ("SOURCE_TREE_SHA1", self.tree), ("FILES", files)):
            patched = patch.object(prepare, attribute, value)
            patched.start()
            self.addCleanup(patched.stop)

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.source), *args], text=True, stderr=subprocess.PIPE)

    def run_prepare(self):
        return prepare.prepare(self.cache, self.source, self.output)

    def test_private_copy_full_hashes_and_independent_verification(self):
        before = {name: (path.read_bytes(), path.stat().st_mode, path.stat().st_mtime_ns) for name, path in self.originals.items()}
        record = self.run_prepare()
        self.assertEqual(record["status"], "PASS")
        self.assertTrue(record["validation"]["all_checkpoint_bytes_hashed"])
        self.assertFalse(record["validation"]["checkpoint_deserialized"])
        self.assertEqual(record["source"]["git_tree_sha1"], self.tree)
        self.assertEqual(len(record["source"]["files"]), 5)
        self.assertTrue((self.output / "pytorch_model.bin").is_symlink())
        self.assertFalse((self.output / "config.json").is_symlink())
        self.assertFalse((self.output / "source/.git").exists())
        self.assertFalse((self.output / "token_bytes.pt").exists())
        self.assertEqual(before, {name: (path.read_bytes(), path.stat().st_mode, path.stat().st_mtime_ns) for name, path in self.originals.items()})
        self.assertEqual(self.git("status", "--porcelain"), "")
        # Verification of the self-contained private copy needs no original Git checkout.
        shutil.rmtree(self.source)
        self.assertEqual(prepare.verify_prepared(self.output), record)

    def test_existing_and_shared_output_paths_are_refused(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.run_prepare()
        self.output.rmdir()
        self.output.symlink_to(self.root / "missing")
        with self.assertRaises(FileExistsError):
            self.run_prepare()
        self.output.unlink()
        for destination in (self.cache / "new", self.source / "new"):
            with self.assertRaises(ValueError):
                prepare.prepare(self.cache, self.source, destination)
            self.assertFalse(destination.exists())

    def test_changed_weight_config_or_tokenizer_is_rejected_before_output(self):
        for name, path in self.originals.items():
            original = path.read_bytes()
            path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                self.run_prepare()
            self.assertFalse(self.output.exists())
            path.write_bytes(original)
        self.originals["pytorch_model.bin"].write_bytes(b"truncated")
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            self.run_prepare()

    def test_wrong_commit_dirty_tracked_and_untracked_source_rejected(self):
        with patch.object(prepare, "SOURCE_REVISION", "0" * 40), self.assertRaisesRegex(ValueError, "exactly"):
            self.run_prepare()
        tracked = self.source / "README.md"
        original = tracked.read_bytes()
        tracked.write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "modified"):
            self.run_prepare()
        tracked.write_bytes(original)
        (self.source / "untracked.py").write_text("raise AssertionError")
        with self.assertRaisesRegex(ValueError, "untracked"):
            self.run_prepare()
        self.assertFalse(self.output.exists())

    def test_assume_unchanged_cannot_hide_changed_source(self):
        self.git("update-index", "--assume-unchanged", "parcae_lm/__init__.py")
        (self.source / "parcae_lm/__init__.py").write_text("print('changed hidden source')\n")
        self.assertEqual(self.git("status", "--porcelain"), "")
        with self.assertRaisesRegex(ValueError, "pinned Git tree"):
            self.run_prepare()
        self.assertFalse(self.output.exists())

    def test_partial_copy_cleanup_preserves_failure_evidence(self):
        real_copy = shutil.copyfile
        def fail_source(src, dst):
            if "source" in Path(dst).relative_to(self.output).parts:
                Path(dst).write_bytes(b"partial")
                raise OSError("simulated disk failure")
            return real_copy(src, dst)
        with patch.object(prepare.shutil, "copyfile", side_effect=fail_source), self.assertRaisesRegex(OSError, "simulated disk"):
            self.run_prepare()
        self.assertEqual({path.name for path in self.output.iterdir()}, {"preparation_failure.json"})
        failure = json.loads((self.output / "preparation_failure.json").read_text())
        self.assertEqual(failure["status"], "FAIL")
        self.assertEqual(failure["phase"], "copy_official_source")
        self.assertEqual(failure["cleanup_errors"], [])
        with self.assertRaises(FileExistsError):
            self.run_prepare()

    def test_verifier_rejects_modified_resigned_extra_or_missing_source(self):
        self.run_prepare()
        target = self.output / "source/README.md"
        original = target.read_bytes()
        manifest_path = self.output / "model_provenance.json"
        manifest_original = manifest_path.read_bytes()
        target.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "fixed Git tree"):
            prepare.verify_prepared(self.output)
        manifest = json.loads(manifest_original)
        manifest["source"]["files"]["README.md"] = {**prepare.stream_fingerprint(target, git_blob=True), "git_mode": "100644"}
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "fixed Git tree"):
            prepare.verify_prepared(self.output)
        target.write_bytes(original)
        manifest_path.write_bytes(manifest_original)
        extra = self.output / "source/injected.py"
        extra.write_text("unexpected")
        with self.assertRaises(ValueError):
            prepare.verify_prepared(self.output)
        extra.unlink()
        target.unlink()
        with self.assertRaises(ValueError):
            prepare.verify_prepared(self.output)

    def test_verifier_rehashes_weight_bytes(self):
        self.run_prepare()
        weight = self.originals["pytorch_model.bin"]
        original = weight.read_bytes()
        weight.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        with self.assertRaisesRegex(ValueError, "Prepared bytes changed"):
            prepare.verify_prepared(self.output)

    def test_hash_reader_uses_only_bounded_chunks(self):
        path = self.root / "bounded.bin"
        data = b"abcdefghijklm"
        path.write_bytes(data)
        calls = []
        class BoundedReader(io.BytesIO):
            def read(self, amount=-1):
                calls.append(amount)
                if amount != 4:
                    raise AssertionError("Unbounded file read")
                return super().read(amount)
        with patch.object(Path, "open", return_value=BoundedReader(data)):
            result = prepare.stream_fingerprint(path, chunk_bytes=4)
        self.assertEqual(result, {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        self.assertEqual(calls, [4] * 5)


if __name__ == "__main__":
    unittest.main()
