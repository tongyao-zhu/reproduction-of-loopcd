"""Package API remains compatible without loading torch for audit utilities."""
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LazyImportTests(unittest.TestCase):
    def isolated(self, code):
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
        result = subprocess.run([sys.executable, "-S", "-c", code], env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_protocol_import_requires_only_standard_library(self):
        self.isolated("""
import importlib.abc, sys
class NoModelDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'numpy', 'transformers'):
            raise AssertionError('Unexpected model dependency: ' + fullname)
sys.meta_path.insert(0, NoModelDependencies())
import loopcd_repro
from loopcd_repro.aime_protocol import sample_seed, PROTOCOL_ID
assert sample_seed('AIME2024-I-01', 0) >= 0
assert PROTOCOL_ID == 'ouro-thinking-aime-reconstruction-v1'
assert 'loopcd_repro.guidance' not in sys.modules
assert 'loopcd_repro.ouro' not in sys.modules
assert loopcd_repro.__version__ == '0.1.0'
assert 'GuidanceConfig' in dir(loopcd_repro)
try:
    loopcd_repro.unknown_attribute
except AttributeError:
    pass
else:
    raise AssertionError('Unknown export accepted')
""")

    def test_existing_public_exports_load_lazily_and_cache_identity(self):
        self.isolated("""
import sys, types
import loopcd_repro
guidance = types.ModuleType('loopcd_repro.guidance')
ouro = types.ModuleType('loopcd_repro.ouro')
for name in ('GuidanceConfig', 'adaptive_strength', 'apply_guidance'):
    setattr(guidance, name, object())
ouro.OuroGuidance = object()
sys.modules[guidance.__name__] = guidance
sys.modules[ouro.__name__] = ouro
assert not any(name in loopcd_repro.__dict__ for name in loopcd_repro.__all__)
from loopcd_repro import GuidanceConfig, OuroGuidance, adaptive_strength, apply_guidance
assert GuidanceConfig is guidance.GuidanceConfig
assert OuroGuidance is ouro.OuroGuidance
assert adaptive_strength is guidance.adaptive_strength
assert apply_guidance is guidance.apply_guidance
assert loopcd_repro.GuidanceConfig is GuidanceConfig
namespace = {}
exec('from loopcd_repro import *', namespace)
assert set(loopcd_repro.__all__) <= set(namespace)
assert namespace['OuroGuidance'] is OuroGuidance
""")


if __name__ == "__main__":
    unittest.main()
