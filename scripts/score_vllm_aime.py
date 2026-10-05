"""Strict complete two-arm vLLM validator and integer-text scorer; executes no answers."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from loopcd_repro.aime_protocol import (load_protocol, load_questions, build_prompt,
    read_model_identity, hash_json, sha256_file, atomic_json)
from loopcd_repro.aime_vllm_protocol import (ARMS, BACKEND_ID, require, validate_batch, scoring_rows)


def score(run, model, project, commit):
    from continue_aime_figure1b import frozen_tree
    from score_aime import load_scoring_data, canonical_gate, summarize_arm, compare_pair
    from transformers import AutoTokenizer
    manifest = json.loads((run/'manifest.json').read_text())
    require(manifest['backend_id'] == BACKEND_ID and manifest['source_commit'] == commit, 'Wrong source/backend')
    require(manifest['source_sha256'] == frozen_tree(ROOT, commit), 'Frozen source changed')
    protocol_path = project/'data/aime/protocol-v1.json'
    data_root = project/'data/aime/prepared-v1'
    protocol = load_protocol(protocol_path)
    identity = read_model_identity(model, protocol)
    require(manifest['model_identity'] == identity, 'Model changed')
    require(manifest['adaptive_cap'] == protocol['models'][identity['repo_id']]['adaptive_cap'], 'Cap changed')
    require(manifest['protocol_sha256'] == sha256_file(protocol_path) and
            manifest['questions_sha256'] == sha256_file(data_root/'aime2024.questions.jsonl'), 'Data changed')
    require(manifest['old_samples_imported'] == 0 and manifest['expected_per_arm'] == 480 and
            manifest['samples_per_problem'] == 16, 'Incomplete protocol')
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, local_files_only=True)
    prompts = [build_prompt(tokenizer, q, protocol) for q in load_questions(data_root, 2024)]
    require(prompts == manifest['prompts'], 'Prompt manifest changed')
    data = load_scoring_data(protocol_path, data_root)
    gate = canonical_gate(data)
    require(json.loads((run/'canonical/canonical_gate.json').read_text()) == gate and
            sha256_file(run/'canonical/canonical_gate.json') == manifest['canonical_sha256'], 'Canonical binding')
    scores, files = {}, {}
    for arm in ARMS:
        batches = []
        expected = [f"{prompt['task_id']}-{i:02d}.json" for prompt in prompts for i in range(0, 16, 2)]
        require(sorted(p.name for p in (run/arm).iterdir()) == sorted(expected), 'Missing or extra batches')
        for prompt in prompts:
            for i in range(0, 16, 2):
                path = run/arm/f"{prompt['task_id']}-{i:02d}.json"
                batch = json.loads(path.read_text())
                validate_batch(batch, manifest, prompt, i, arm, tokenizer)
                files[str(path.relative_to(run))] = sha256_file(path)
                batches.append(batch)
        scores[arm] = summarize_arm(scoring_rows(batches), data, 2024)
    result = dict(status='PASS', backend_id=BACKEND_ID, full_480_per_arm_verified=True,
                  manifest_sha256=sha256_file(run/'manifest.json'), batch_files_sha256=files,
                  canonical_gate=gate, scores=scores, comparison=compare_pair(scores['baseline'], scores['adaptive']),
                  elapsed_policy=manifest['seconds_policy'], limitation=manifest['limitation'])
    require(not (run/'comparison.json').exists(), 'Refusing score overwrite')
    atomic_json(run/'comparison.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('run', 'model', 'project'):
        p.add_argument('--'+n, type=Path, required=True)
    p.add_argument('--source-commit', required=True)
    a = p.parse_args()
    r = score(a.run, a.model, a.project, a.source_commit)
    print(json.dumps(r['comparison']['metrics']))
