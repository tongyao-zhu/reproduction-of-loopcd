"""CPU-only tests for pinned data integrity and generation/gold separation."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_aime_data.py"
SPEC = importlib.util.spec_from_file_location("prepare_aime_data", SCRIPT)
aime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(aime)


def rows2024():
    rows = []
    for part in ("I", "II"):
        for number in sorted(range(1, 16), key=str):
            rows.append({
                "id": 60 + len(rows), "year": "2024", "problem": f"2024 {part} problem {number}",
                "solution": "PRIVATE_SOLUTION_SENTINEL", "answer": "025" if number == 2 else "123",
                "url": f"https://artofproblemsolving.com/wiki/index.php/2024_AIME_{part}_Problems/Problem_{number}",
            })
    return rows


def rows2025():
    return {part: [{"question": f"2025 {part} problem {number}",
                    "answer": r"336^\\circ" if part == "II" and number == 5 else "000"}
                   for number in range(1, 16)] for part in ("I", "II")}


class GoldTests(unittest.TestCase):
    def test_integer_boundaries_and_leading_zeros(self):
        for raw, expected in (("0", 0), ("000", 0), ("025", 25), ("999", 999), (" 073 ", 73)):
            with self.subTest(raw=raw):
                value, events = aime.normalize_gold(raw)
                self.assertEqual(value, expected)
                if raw.strip().startswith("0") and len(raw.strip()) > 1:
                    self.assertIn("remove_leading_zeros", events)

    def test_observed_degree_suffix_is_explicit_and_recorded(self):
        for raw in (r"336^\circ", r"336^\\circ"):
            self.assertEqual(aime.normalize_gold(raw, allow_degree_suffix=True),
                             (336, ["remove_explicit_degree_suffix"]))
            with self.assertRaises(ValueError):
                aime.normalize_gold(raw)

    def test_never_eval_modulo_or_coerce(self):
        for raw in (True, 25, 25.0, "-1", "+1", "1000", "0000", "1.0", "25/1", "1+2", "３３６", "", "NaN", "336 degrees", r"\boxed{336}"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                aime.normalize_gold(raw, allow_degree_suffix=True)


class RecordTests(unittest.TestCase):
    def test_2024_numeric_order_with_original_metadata(self):
        pairs = aime.normalize_aime2024(rows2024())
        self.assertEqual([q["problem_number"] for q, _ in pairs[:15]], list(range(1, 16)))
        second, answer = pairs[1]
        self.assertEqual(second["task_id"], "AIME2024-I-02")
        self.assertEqual(second["source"]["row"], 8)
        self.assertEqual(second["source"]["id"], 67)
        self.assertEqual(answer["gold_raw"], "025")
        self.assertEqual(answer["gold_int"], 25)
        self.assertNotIn("solution", second)
        self.assertNotIn("answer", second)

    def test_2025_both_exams_and_degree_only_on_known_record(self):
        pairs = aime.normalize_aime2025(rows2025())
        self.assertEqual(len(pairs), 30)
        question, answer = pairs[19]
        self.assertEqual(question["task_id"], "AIME2025-II-05")
        self.assertEqual(question["source"]["row"], 5)
        self.assertEqual(question["source"]["problem_number_origin"], "source_file_row_1_based")
        self.assertEqual(answer["gold_int"], 336)
        rows = rows2025()
        rows["I"][0]["answer"] = r"336^\\circ"
        with self.assertRaises(ValueError):
            aime.normalize_aime2025(rows)

    def test_question_bytes_are_preserved(self):
        rows = rows2025()
        original = "  Source math: \\frac{1}{2}.\nSecond line.  "
        rows["I"][0]["question"] = original
        question, _ = aime.normalize_aime2025(rows)[0]
        self.assertEqual(question["question"], original)
        self.assertEqual(question["question_sha256"], aime.digest(original.encode("utf-8")))

    def test_reject_partial_exam(self):
        with self.assertRaises(ValueError):
            aime.normalize_aime2024(rows2024()[:-1])
        for parts in ({"I": rows2025()["I"]}, {"I": rows2025()["I"], "II": rows2025()["II"][:-1]}):
            with self.assertRaises(ValueError):
                aime.normalize_aime2025(parts)

    def test_reject_duplicate_missing_and_out_of_range_2024_problem(self):
        for replacement in ("Problem_1", "Problem_16"):
            rows = rows2024()
            rows[1]["url"] = rows[0]["url"].replace("Problem_1", replacement)
            with self.assertRaises(ValueError):
                aime.normalize_aime2024(rows)

    def test_reject_changed_source_identity(self):
        for field, value in (("id", 0), ("year", "2023"), ("url", "missing-exam-label")):
            rows = rows2024()
            rows[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                aime.normalize_aime2024(rows)

    def test_reject_duplicate_question_text(self):
        rows = rows2025()
        rows["II"][0]["question"] = rows["I"][0]["question"]
        with self.assertRaises(ValueError):
            aime.normalize_aime2025(rows)

    def test_reject_schema_drift_and_empty_questions(self):
        for key, value in (("solution", "unexpected"), ("question", " \n")):
            rows = rows2025()
            rows["I"][0][key] = value
            with self.assertRaises(ValueError):
                aime.normalize_aime2025(rows)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw, self.output = self.root / "raw", self.root / "prepared"
        self.sources = deepcopy(aime.SOURCES)
        fixtures = {"aime2024": {"README.md": b"2024 source card", "data/train-00000-of-00001.parquet": b"parquet fixture"},
                    "aime2025": {"README.md": b"2025 source card", **{
                        f"aime2025-{part}.jsonl": ("\n".join(json.dumps(row) for row in rows) + "\n").encode()
                        for part, rows in rows2025().items()}}}
        for dataset, files in fixtures.items():
            for filename, data in files.items():
                path = aime.raw_directory(self.raw, self.sources[dataset]) / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                self.sources[dataset]["files"][filename] = {"bytes": len(data), "sha256": aime.digest(data)}
        self.addCleanup(patch.stopall)
        patch.object(aime, "SOURCES", self.sources).start()
        patch.object(aime, "parse_parquet", return_value=rows2024()).start()

    def test_complete_prepare_with_verified_separate_files(self):
        manifest = aime.prepare(self.raw, self.output)
        self.assertEqual(manifest["status"], "PASS")
        self.assertEqual(manifest["source_script_sha256"], aime.digest(SCRIPT.read_bytes()))
        for year in (2024, 2025):
            dataset = f"aime{year}"
            evidence = manifest["datasets"][dataset]
            self.assertEqual(evidence["parts"], {"I": 15, "II": 15})
            question_bytes = (self.output / evidence["files"]["questions"]["file"]).read_bytes()
            questions = aime.parse_jsonl(question_bytes)
            answers = aime.parse_jsonl((self.output / evidence["files"]["answers"]["file"]).read_bytes())
            self.assertEqual(len(questions), 30)
            self.assertEqual({q["task_id"] for q in questions}, {a["task_id"] for a in answers})
            self.assertNotIn(b"PRIVATE_SOLUTION_SENTINEL", question_bytes)
            for question in questions:
                self.assertFalse(set(question) & {"answer", "solution", "gold_raw", "gold_int"})
            for file in evidence["files"].values():
                data = (self.output / file["file"]).read_bytes()
                self.assertEqual(len(data), file["bytes"])
                self.assertEqual(aime.digest(data), file["sha256"])
        events = manifest["datasets"]["aime2025"]["normalization_events"]
        self.assertEqual(sum("remove_explicit_degree_suffix" in event["normalization"] for event in events), 1)

    def test_hash_tamper_refused_before_output_is_created(self):
        path = aime.raw_directory(self.raw, self.sources["aime2025"]) / "aime2025-I.jsonl"
        data = path.read_bytes()
        path.write_bytes(data.replace(b"000", b"001", 1))
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            aime.prepare(self.raw, self.output)
        self.assertFalse(self.output.exists())

    def test_existing_output_and_dangling_symlink_refused(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            aime.prepare(self.raw, self.output)
        self.output.rmdir()
        self.output.symlink_to(self.root / "missing")
        with self.assertRaises(FileExistsError):
            aime.prepare(self.raw, self.output)

    def test_no_download_by_default(self):
        with patch.object(aime.urllib.request, "urlopen", side_effect=AssertionError("network forbidden")):
            aime.prepare(self.raw, self.output)

    def test_download_existing_pins_is_idempotent_and_never_overwrites_changed_file(self):
        with patch.object(aime.urllib.request, "urlopen", side_effect=AssertionError("network forbidden")):
            aime.download_raw(self.raw)
        path = aime.raw_directory(self.raw, self.sources["aime2024"]) / "README.md"
        path.write_bytes(b"changed")
        with self.assertRaises(ValueError):
            aime.download_raw(self.raw)
        self.assertEqual(path.read_bytes(), b"changed")

    def test_shared_cache_output_rejected_including_resolved_symlink(self):
        cache = self.root / ".cache" / "huggingface" / "hub"
        cache.mkdir(parents=True)
        linked = self.root / "linked"
        linked.symlink_to(cache, target_is_directory=True)
        for target in (cache / "new", linked / "new", self.root / "datasets--example--source" / "new"):
            with self.subTest(target=target), self.assertRaises(ValueError):
                aime.prepare(self.raw, target)
            self.assertFalse(target.exists())

    def test_output_records_are_deterministic(self):
        first = aime.prepare(self.raw, self.output)
        second_output = self.root / "second"
        second = aime.prepare(self.raw, second_output)
        self.assertEqual(first["datasets"], second["datasets"])
        for filename in self.output.glob("*.jsonl"):
            self.assertEqual(filename.read_bytes(), (second_output / filename.name).read_bytes())

    def test_source_is_not_modified(self):
        before = {str(p): p.read_bytes() for p in self.raw.rglob("*") if p.is_file()}
        aime.prepare(self.raw, self.output)
        after = {str(p): p.read_bytes() for p in self.raw.rglob("*") if p.is_file()}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
