import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from compare_huginn import trace_evidence


class TraceComparisonTests(unittest.TestCase):
    def record(self):
        return {"index": 0, "input_ids_sha256": "tokenhash", "input_shape": [1, 4],
                "rng_before": {"cpu": "a", "model_device": "b"},
                "rng_after": {"cpu": "a", "model_device": "c"},
                "initialization": {"rng_before": {"cpu": "a", "model_device": "b"},
                                   "rng_after": {"cpu": "a", "model_device": "c"},
                                   "scale": 1.0, "first_16_values_sha256": "noisehash"}}

    def evidence(self, directory, record, count=1):
        (Path(directory) / "request_trace.jsonl").write_text(json.dumps(record) + "\n")
        return trace_evidence(directory, {"request_audit": {"complete": True, "request_count": count, "planned_requests": count}})

    def test_trace_binds_actual_inputs_and_native_noise(self):
        with tempfile.TemporaryDirectory() as directory:
            record = self.record()
            initial = self.evidence(directory, record)
            record["initialization"]["first_16_values_sha256"] = "othernoise"
            changed = self.evidence(directory, record)
            self.assertNotEqual(initial["trace_sha256"], changed["trace_sha256"])

    def test_missing_record_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                self.evidence(directory, self.record(), count=2)

    def test_extra_random_draw_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            record = self.record()
            record["rng_after"]["model_device"] = "extra-random-draw"
            with self.assertRaises(ValueError):
                self.evidence(directory, record)


if __name__ == "__main__":
    unittest.main()
