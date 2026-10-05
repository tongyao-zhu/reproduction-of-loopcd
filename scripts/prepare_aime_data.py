"""Prepare pinned AIME sources into separate generation and scoring inputs.

No model, tokenizer, Hub cache, or GPU is touched. Offline by default; --download
fetches only the five small, hash-pinned public files into a private raw directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import urllib.request


SOURCES = {
    "aime2024": {
        "repo_id": "HuggingFaceH4/aime_2024",
        "revision": "2fe88a2f1091d5048c0f36abc874fb997b3dd99a",
        "split": "train",
        "files": {
            "README.md": {"bytes": 932, "sha256": "c1d82f7870ba18b505fad3027c5c6131bac989ed3e177e9fc23385b9b9b315b8"},
            "data/train-00000-of-00001.parquet": {
                "bytes": 81670,
                "sha256": "26139847601a5037c237d5928b195e7260ca8074cf4f264b794af42847f79ccf",
            },
        },
    },
    "aime2025": {
        "repo_id": "opencompass/AIME2025",
        "revision": "a6ad95f611d72cf628a80b58bd0432ef6638f958",
        "split": "test",
        "files": {
            "README.md": {"bytes": 448, "sha256": "43ac9ef26311be77671372031a242d031858ba836a6d79f323a1bac748e012ac"},
            "aime2025-I.jsonl": {"bytes": 5764, "sha256": "b91b3c96f05d9635d2a0692b124ebe023c1ff59cb19c074275e6c4b349d0659e"},
            "aime2025-II.jsonl": {"bytes": 6285, "sha256": "16a2dcfbbf9db1b11f8a69a3ba5e4cac73e3641b19a37e2307e9c12240bbed5e"},
        },
    },
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical_json(data):
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def raw_directory(raw_root, spec):
    return Path(raw_root) / spec["repo_id"].replace("/", "--") / spec["revision"]


def source_url(spec, filename):
    return f"https://huggingface.co/datasets/{spec['repo_id']}/resolve/{spec['revision']}/{filename}"


def assert_private_directory(path):
    resolved = Path(path).expanduser().resolve()
    parts = resolved.parts
    if ("huggingface" in parts and "hub" in parts) or any(
        part.startswith(("datasets--", "models--")) for part in parts
    ):
        raise ValueError("Private preparation output must not be inside a shared Hugging Face Hub cache")
    return resolved


def verify_bytes(data, expected, label):
    if len(data) != expected["bytes"] or digest(data) != expected["sha256"]:
        raise ValueError(f"Pinned source bytes/hash mismatch: {label}")


def download_raw(raw_root):
    """Download missing raw files; existing files must already match the pin."""
    raw_root = assert_private_directory(raw_root)
    for spec in SOURCES.values():
        for filename, expected in spec["files"].items():
            path = raw_directory(raw_root, spec) / filename
            if path.is_symlink():
                raise ValueError(f"Refusing to write through a raw-file symlink: {path}")
            if path.exists():
                verify_bytes(path.read_bytes(), expected, str(path))
                continue
            # Only official HTTPS dataset URLs at an immutable commit are used.
            with urllib.request.urlopen(source_url(spec, filename), timeout=60) as response:
                data = response.read(expected["bytes"] + 1)
            verify_bytes(data, expected, source_url(spec, filename))
            assert_private_directory(path.parent)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(data)


def normalize_gold(value, *, allow_degree_suffix=False):
    """Normalize source gold only; this is not a generated-answer extractor.

    The pinned OpenCompass AIME II problem 5 has a degree suffix. Its raw value is
    retained, and the narrowly recognized suffix is recorded rather than evaluated.
    """
    if not isinstance(value, str):
        raise ValueError("AIME source answer must be a string, not a numeric coercion")
    text = value.strip()
    changes = ["strip_outer_whitespace"] if text != value else []
    if allow_degree_suffix:
        match = re.fullmatch(r"([0-9]{1,3})\^\\{1,2}circ", text)
        if match:
            text = match.group(1)
            changes.append("remove_explicit_degree_suffix")
    if not re.fullmatch(r"[0-9]{1,3}", text):
        raise ValueError(f"Not an AIME integer in 0..999: {value!r}")
    result = int(text, 10)
    if len(text) > 1 and text.startswith("0"):
        changes.append("remove_leading_zeros")
    return result, changes


def make_pair(year, part, number, question, gold, source):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be nonempty source text")
    task_id = f"AIME{year}-{part}-{number:02d}"
    gold_int, changes = normalize_gold(
        gold, allow_degree_suffix=(year == 2025 and part == "II" and number == 5)
    )
    generation = {
        "task_id": task_id, "dataset": f"aime{year}", "year": year,
        "part": part, "problem_number": number, "question": question,
        "question_sha256": digest(question.encode("utf-8")), "source": source,
    }
    answer = {
        "task_id": task_id, "question_sha256": generation["question_sha256"],
        "gold_raw": gold, "gold_int": gold_int, "normalization": changes,
    }
    return generation, answer


def normalize_aime2024(rows):
    """Use the original URL's exam/number, not its lexicographic row order."""
    spec = SOURCES["aime2024"]
    if len(rows) != 30:
        raise ValueError("AIME 2024 requires all 30 source rows")
    pairs, original_ids = [], set()
    for row_number, row in enumerate(rows, 1):
        if not {"id", "problem", "solution", "answer", "url", "year"} <= set(row):
            raise ValueError("Unexpected AIME 2024 schema")
        if str(row["year"]) != "2024":
            raise ValueError("Wrong AIME source year")
        if type(row["id"]) is not int or row["id"] in original_ids:
            raise ValueError("AIME 2024 source IDs must be unique integers")
        original_ids.add(row["id"])
        match = re.search(r"(?:^|/)2024_AIME_(I|II)_Problems/Problem_([0-9]{1,2})$", row["url"])
        if not match:
            raise ValueError("AIME 2024 source URL must identify exam and problem")
        part, number = match.group(1), int(match.group(2))
        source = {
            "repo_id": spec["repo_id"], "revision": spec["revision"], "split": "train",
            "file": "data/train-00000-of-00001.parquet", "row": row_number,
            "id": row["id"], "url": row["url"], "problem_number_origin": "source_url",
        }
        pairs.append(make_pair(2024, part, number, row["problem"], row["answer"], source))
    if original_ids != set(range(60, 90)):
        raise ValueError("Pinned AIME 2024 must retain original IDs 60..89")
    return validate_pairs(pairs, 2024)


def normalize_aime2025(parts):
    spec = SOURCES["aime2025"]
    if set(parts) != {"I", "II"}:
        raise ValueError("AIME 2025 requires both exam parts")
    pairs = []
    for part in ("I", "II"):
        if len(parts[part]) != 15:
            raise ValueError(f"AIME 2025 {part} requires all 15 source rows")
        for row_number, row in enumerate(parts[part], 1):
            if set(row) != {"question", "answer"}:
                raise ValueError("Unexpected AIME 2025 schema")
            source = {
                "repo_id": spec["repo_id"], "revision": spec["revision"], "split": "test",
                "config": f"AIME2025-{part}", "file": f"aime2025-{part}.jsonl",
                "row": row_number, "problem_number_origin": "source_file_row_1_based",
            }
            pairs.append(make_pair(2025, part, row_number, row["question"], row["answer"], source))
    return validate_pairs(pairs, 2025)


def validate_pairs(pairs, year):
    expected = {f"AIME{year}-{part}-{number:02d}" for part in ("I", "II") for number in range(1, 16)}
    ids = [question["task_id"] for question, _ in pairs]
    hashes = [question["question_sha256"] for question, _ in pairs]
    if len(pairs) != 30 or set(ids) != expected or len(set(ids)) != 30:
        raise ValueError("Dataset must contain exactly I/II problems 1..15 once each")
    if len(set(hashes)) != 30:
        raise ValueError("Duplicate question text in dataset")
    for question, answer in pairs:
        if answer["task_id"] != question["task_id"] or answer["question_sha256"] != question["question_sha256"]:
            raise ValueError("Question and gold identity mismatch")
    return sorted(pairs, key=lambda pair: (pair[0]["part"], pair[0]["problem_number"]))


def parse_parquet(data):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("AIME 2024 requires pyarrow; use the existing CPU evaluation environment") from exc
    return pq.read_table(io.BytesIO(data)).to_pylist()


def parse_jsonl(data):
    lines = data.decode("utf-8").splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError("Empty line or empty source JSONL")
    return [json.loads(line) for line in lines]


def prepare(raw_root, output):
    raw_root, output = Path(raw_root).expanduser().resolve(), Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite prepared data: {output}")
    assert_private_directory(output)
    source_bytes, sources_manifest = {}, {}
    # Validate every byte before parsing or creating output. Parsing uses those
    # same bytes, so a source change between validation and read cannot slip in.
    for name, spec in SOURCES.items():
        source_bytes[name], files = {}, {}
        for filename, expected in spec["files"].items():
            path = raw_directory(raw_root, spec) / filename
            data = path.read_bytes()
            verify_bytes(data, expected, str(path))
            source_bytes[name][filename] = data
            files[filename] = {**expected, "url": source_url(spec, filename)}
        sources_manifest[name] = {
            "repo_id": spec["repo_id"], "revision": spec["revision"],
            "split": spec["split"], "files": files,
        }
    pairs_by_dataset = {
        "aime2024": normalize_aime2024(parse_parquet(source_bytes["aime2024"]["data/train-00000-of-00001.parquet"])),
        "aime2025": normalize_aime2025({
            part: parse_jsonl(source_bytes["aime2025"][f"aime2025-{part}.jsonl"])
            for part in ("I", "II")
        }),
    }
    output_bytes, datasets_manifest = {}, {}
    for dataset, pairs in pairs_by_dataset.items():
        files = {}
        for label, index in (("questions", 0), ("answers", 1)):
            filename = f"{dataset}.{label}.jsonl"
            data = ("\n".join(canonical_json(pair[index]) for pair in pairs) + "\n").encode("utf-8")
            output_bytes[filename] = data
            files[label] = {"file": filename, "sha256": digest(data), "bytes": len(data), "rows": 30}
        datasets_manifest[dataset] = {
            "rows": 30, "parts": {"I": 15, "II": 15}, "files": files,
            "task_ids": [question["task_id"] for question, _ in pairs],
            "normalization_events": [
                {"task_id": answer["task_id"], "gold_raw": answer["gold_raw"],
                 "gold_int": answer["gold_int"], "normalization": answer["normalization"]}
                for _, answer in pairs if answer["normalization"]
            ],
        }
    manifest = {
        "schema_version": 1, "status": "PASS", "kind": "pinned_aime_data_preparation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_script_sha256": digest(Path(__file__).read_bytes()),
        "sources": sources_manifest, "datasets": datasets_manifest,
        "validation": {"source_bytes_verified": True, "exact_exam_problem_sets": True,
                       "unique_questions_per_year": True, "gold_range_0_999": True,
                       "generation_files_contain_no_gold_or_solutions": True},
        "ordering": "year, exam I then II, numeric problem number 1..15",
        "scope": "Data preparation only; no model loaded, generated solution, or score.",
    }
    output.mkdir(parents=True, exist_ok=False)
    for filename, data in output_bytes.items():
        with (output / filename).open("xb") as handle:
            handle.write(data)
    with (output / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/aime/raw"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true", help="Download only hash-pinned raw files to --raw-root")
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite prepared data: {args.output}")
    assert_private_directory(args.output)
    if args.download:
        download_raw(args.raw_root)
    result = prepare(args.raw_root, args.output)
    print(json.dumps({"status": result["status"], "output": str(args.output),
                      "counts": {name: info["rows"] for name, info in result["datasets"].items()}}, sort_keys=True))


if __name__ == "__main__":
    main()
