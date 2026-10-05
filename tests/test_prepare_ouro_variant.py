"""Fail closed on incomplete/wrong snapshots before creating private model files."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "prepare_ouro_variant", Path(__file__).resolve().parents[1] / "scripts/prepare_ouro_variant.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fixture(self, name="Ouro-2.6B"):
        spec = copy.deepcopy(module.VARIANTS[name])
        source = self.root / "hub" / ("models--ByteDance--" + name) / "snapshots" / spec["revision"]
        source.mkdir(parents=True)
        code = ("class UniversalTransformerCache:\n" + module.MARKER
                + "\n        self._seen_tokens = 0\n")
        if spec["native_mask_sizes"]:
            code += "\n    def get_mask_sizes(self, cache_position, layer_idx=0):\n        return (91, 0)\n"
        for file in spec["audited_files"]:
            (source / file).write_text("{}\n")
        (source / "modeling_ouro.py").write_text(code)
        config = {"architectures": ["OuroForCausalLM"], "model_type": "ouro",
                  "num_hidden_layers": spec["layers"], "hidden_size": 2048,
                  "total_ut_steps": 4, "early_exit_threshold": 1.0}
        (source / "config.json").write_text(json.dumps(config))
        blob = self.root / (name + ".weight_blob")
        blob.write_bytes(b"test model weights")
        (source / "model.safetensors").symlink_to(blob)
        spec["weights_bytes"] = blob.stat().st_size
        spec["audited_files"] = {file: module.sha256(source / file) for file in spec["audited_files"]}
        self.addCleanup(patch.stopall)
        patch.dict(module.VARIANTS, {name: spec}).start()
        return source, self.root / ("private-" + name), spec

    def test_all_four_sources_preserved_and_weight_files_only_linked(self):
        for name in tuple(module.VARIANTS):
            with self.subTest(model=name):
                source, output, spec = self.fixture(name)
                before = {p.name: p.read_bytes() for p in source.iterdir()}
                record = module.prepare(source, output, name)
                self.assertEqual({p.name: p.read_bytes() for p in source.iterdir()}, before)
                self.assertEqual(record["repo_id"], "ByteDance/" + name)
                self.assertEqual(record["revision"], spec["revision"])
                self.assertTrue((output / "model.safetensors").is_symlink())
                self.assertEqual((output / "model.safetensors").resolve(), (source / "model.safetensors").resolve())
                self.assertFalse((output / "config.json").is_symlink())
                self.assertEqual((output / "config.json").read_bytes(), before["config.json"])
                code = (output / "modeling_ouro.py").read_text()
                self.assertEqual(code.count("    def get_mask_sizes("), 1)
                self.assertIn("key_cache = None", code)
                if spec["native_mask_sizes"]:
                    self.assertIn("return (91, 0)", code)
                    self.assertNotIn("self._seen_tokens + new_tokens", code)
                else:
                    self.assertIn("self._seen_tokens + new_tokens", code)
                self.assertEqual(record["model_code_sha256"], module.sha256(output / "modeling_ouro.py"))
                self.assertEqual(record["compatibility_patch_sha256"], hashlib.sha256(record["compatibility_patch_diff"].encode()).hexdigest())
                self.assertFalse(record["validation"]["weight_content_rehashed"])
                self.assertTrue(record["validation"]["real_model_smoke_required"])

    def test_changed_tokenizer_fails_before_output_creation(self):
        source, output, _ = self.fixture()
        (source / "tokenizer_config.json").write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, "tokenizer_config.json"):
            module.prepare(source, output, "Ouro-2.6B")
        self.assertFalse(output.exists())

    def test_different_model_source_is_rejected(self):
        source, output, _ = self.fixture()
        with self.assertRaisesRegex(ValueError, "exact pinned"):
            module.prepare(source, output, "Ouro-1.4B")
        self.assertFalse(output.exists())

    def test_missing_and_truncated_weights_rejected(self):
        for missing in (True, False):
            with self.subTest(missing=missing):
                source, output, _ = self.fixture("Ouro-1.4B" if missing else "Ouro-2.6B")
                weights = source / "model.safetensors"
                if missing:
                    weights.resolve().unlink()
                else:
                    weights.write_bytes(b"short")
                with self.assertRaisesRegex(ValueError, "checkpoint is missing or incomplete"):
                    module.prepare(source, output, "Ouro-1.4B" if missing else "Ouro-2.6B")
                self.assertFalse(output.exists())

    def test_existing_destination_and_dangling_symlink_rejected(self):
        source, output, _ = self.fixture()
        output.symlink_to(self.root / "missing")
        with self.assertRaises(FileExistsError):
            module.prepare(source, output, "Ouro-2.6B")
        self.assertTrue(output.is_symlink())

    def test_shared_cache_destination_rejected(self):
        source, _, _ = self.fixture()
        output = self.root / "hub" / "private-model"
        with self.assertRaisesRegex(ValueError, "outside the shared"):
            module.prepare(source, output, "Ouro-2.6B")
        self.assertFalse(output.exists())

    def test_native_interface_not_accidentally_patched_twice(self):
        source, output, spec = self.fixture()
        p = source / "modeling_ouro.py"
        p.write_text(p.read_text() + "\n    def get_mask_sizes(self, x, layer_idx=0):\n        return (1, 0)\n")
        spec["audited_files"][p.name] = module.sha256(p)
        with self.assertRaisesRegex(ValueError, "mask interface"):
            module.prepare(source, output, "Ouro-2.6B")
        self.assertFalse(output.exists())

    def test_unexpected_architecture_rejected_even_with_file_hash(self):
        source, output, spec = self.fixture()
        p = source / "config.json"
        config = json.loads(p.read_text())
        config["total_ut_steps"] = 3
        p.write_text(json.dumps(config))
        spec["audited_files"][p.name] = module.sha256(p)
        with self.assertRaisesRegex(ValueError, "total_ut_steps"):
            module.prepare(source, output, "Ouro-2.6B")
        self.assertFalse(output.exists())

    def test_extra_weight_or_dangling_ancillary_file_rejected(self):
        for name, dangling in (("Ouro-1.4B", False), ("Ouro-2.6B", True)):
            source, output, _ = self.fixture(name)
            if dangling:
                (source / "README.md").symlink_to(self.root / "missing-readme")
            else:
                (source / "pytorch_model.bin").write_bytes(b"unknown checkpoint")
            with self.assertRaises(ValueError):
                module.prepare(source, output, name)
            self.assertFalse(output.exists())

    def test_offline_resolution_never_needs_hub_import_or_network(self):
        source, _, _ = self.fixture()
        self.assertEqual(module.resolve_snapshot("Ouro-2.6B", self.root / "hub"), source.resolve())
        with self.assertRaises(FileNotFoundError):
            module.resolve_snapshot("Ouro-1.4B", self.root / "hub")


if __name__ == "__main__":
    unittest.main()
