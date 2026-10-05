"""Strict two-arm AIME2024 scoring with frozen engine/scheduler lineage.

Only parses generated text. --audit-only validates partial files without
extracting answers or reporting benchmark scores. No generated code executes.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import score_aime as s
from continue_aime_figure1b import frozen_tree, sha, ENGINE_COMMIT
from loopcd_repro.aime_protocol import validate_smoke, smoke_bindings

DRIVER_COMMIT = '533ce575969156fc4302901624ca0ef67aedcc9b'
ARMS = ('baseline', 'adaptive')
PAIR_FIELDS = ('task_id', 'sample_id', 'seed', 'question_sha256', 'messages', 'messages_sha256',
               'prompt', 'prompt_sha256', 'prompt_token_ids', 'prompt_token_ids_sha256',
               'generation_config_sha256', 'effective_max_new_tokens')


def source_evidence():
    return {str(p.relative_to(ROOT)): sha(p) for p in (Path(__file__), ROOT/'scripts/score_aime.py',
            ROOT/'scripts/continue_aime_figure1b.py', ROOT/'src/loopcd_repro/aime_protocol.py')}


def partial_generation(folder, data):
    """Same independent row checks as full scoring, permitting ordered prefixes."""
    manifest, manifest_sha = s.json_snapshot(folder/'manifest.json')
    s.same(manifest['kind'], 'aime_generation', 'generation kind')
    if manifest.get('status') not in ('running', 'completed') or manifest.get('error'):
        raise ValueError('Generation failed or has unknown status')
    s.same(manifest['is_full_split'], True, 'full target')
    s.same(manifest['expected_samples'], 480, 'expected 480')
    config = manifest['config']
    year, arm = s.validate_configuration(config, data)
    s.same(year, 2024, 'Figure1b year')
    common = {k:v for k,v in config.items() if k != 'guidance'}
    s.same(manifest['config_hash'], s.hash_json(config), 'config hash')
    s.same(manifest['paired_config_hash'], s.hash_json(common), 'paired config hash')
    raw, checksum = s.snapshot(folder/'samples.jsonl')
    if raw and not raw.endswith(b'\n'):
        raise ValueError('Unterminated sample file')
    rows = [json.loads(line) for line in raw.splitlines()]
    s.same(len(rows), manifest['completed_samples'], 'stable completed count')
    if len(rows) > 480:
        raise ValueError('Too many rows')
    keys = [(task, i) for task in s.task_ids(2024) for i in range(16)]
    for row, key in zip(rows, keys):
        s.validate_row(row, config, data, key)
    if manifest['status'] == 'completed':
        # Includes exactly480, cap totals and final sample hash validation.
        s.read_generation(folder, data)
    return dict(path=str(folder), manifest=manifest, manifest_sha256=manifest_sha,
                config=config, common_config=common, arm=arm, year=year, rows=rows, samples_sha256=checksum, raw=raw)


def audit(folder, protocol, data_root, engine_root, driver_root, smoke, full=False, predecessor=None):
    folder = Path(folder)
    data = s.load_scoring_data(protocol, data_root)
    gate = s.canonical_gate(data)
    engine_pins = frozen_tree(Path(engine_root), ENGINE_COMMIT)
    driver_pins = frozen_tree(Path(driver_root), DRIVER_COMMIT)
    continuation, continuation_sha = s.json_snapshot(folder/'continuation.json')
    b = continuation['binding']
    for key, expected in {'schema_version':1, 'kind':'figure1b_aime2024_continuation', 'engine_commit':ENGINE_COMMIT,
                          'engine_source_sha256':engine_pins, 'driver_commit':DRIVER_COMMIT,
                          'driver_source_sha256':driver_pins, 'arms':list(ARMS), 'year':2024, 'samples_per_arm':480,
                          'imported_sample_bytes_unchanged':True, 'generation_parameters_unchanged':True}.items():
        s.same(b[key], expected, 'continuation '+key)
    bound_sha = s.hash_json(b)
    old_root = Path(predecessor or b['predecessor'])
    runs = {a:partial_generation(folder/a, data) for a in ARMS}
    baseline = runs['baseline']
    if full:
        state, _ = s.json_snapshot(folder/'status.json')
        s.same(state['status'], 'completed_generation_unscored', 'completed driver')
        s.same(state['binding_sha256'], bound_sha, 'driver final binding')
        s.same(state['completed'], {a:480 for a in ARMS}, 'driver final counts')
    for arm, run in runs.items():
        s.same(run['arm'], arm, 'arm directory')
        s.same(run['common_config'], baseline['common_config'], 'paired model/runtime/source/protocol')
        source=run['config']['source']
        s.same(source['git_commit'], ENGINE_COMMIT, 'actual engine commit')
        s.same(source['source_sha256'], engine_pins, 'actual engine tree')
        s.same(run['manifest']['execution_driver'], {'commit':DRIVER_COMMIT, 'continuation_binding_sha256':bound_sha}, 'driver manifest')
        s.same(run['manifest']['gpu_smoke_sha256'], sha(smoke), 'actual smoke bytes')
        validate_smoke(smoke, smoke_bindings(run['config']['model_identity'], source))
        if full:
            s.read_generation(folder/arm, data)
    if not (len(runs['baseline']['rows']) >= len(runs['adaptive']['rows']) and
            len(runs['baseline']['rows']) - len(runs['adaptive']['rows']) <= 1):
        raise ValueError('Not an interleaved two-arm checkpoint')
    for before, after in zip(baseline['rows'], runs['adaptive']['rows']):
        for key in PAIR_FIELDS:
            s.same(before[key], after[key], 'paired row '+key)
    s.same(sorted(b['predecessor_files']), ['adaptive', 'baseline', 'fixed'], 'original three-arm set')
    imported = {}
    for arm in ('baseline', 'fixed', 'adaptive'):
        path=old_root/arm
        proof=b['predecessor_files'][arm]
        old_manifest, manifest_sha=s.json_snapshot(path/'manifest.json')
        raw, raw_sha=s.snapshot(path/'samples.jsonl')
        s.same(manifest_sha, proof['manifest_sha256'], 'unchanged old manifest')
        s.same(raw_sha, proof['samples_sha256'], 'unchanged old samples')
        config=old_manifest['config']
        s.same(s.validate_configuration(config,data), (2024,arm), 'old configuration')
        s.same({k:v for k,v in config.items() if k!='guidance'}, baseline['common_config'], 'old/new paired config')
        s.same(old_manifest['config_hash'], s.hash_json(config), 'old config hash')
        s.same(proof['config_hash'], old_manifest['config_hash'], 'bound old config')
        old_rows=[json.loads(line) for line in raw.splitlines()]
        if len(old_rows) > 480 or (raw and not raw.endswith(b'\n')):
            raise ValueError('Invalid original prefix length or terminator')
        s.same(len(old_rows),proof['n'],'import count')
        s.same(old_manifest['completed_samples'],len(old_rows),'old stable count')
        for row,key in zip(old_rows,[(t,i) for t in s.task_ids(2024) for i in range(16)]):
            s.validate_row(row,config,data,key)
        if arm in ARMS:
            if not runs[arm]['raw'].startswith(raw):
                raise ValueError('Imported row bytes changed')
            s.same(config,runs[arm]['config'],'old/new exact configuration')
        imported[arm]=len(old_rows)
    if not (imported['baseline'] >= imported['fixed'] >= imported['adaptive'] and imported['baseline'] - imported['adaptive'] <= 1):
        raise ValueError('Original arms were not an interleaved checkpoint')
    evidence={'status':'PASS','audit_only':not full,'created_at':datetime.now(timezone.utc).isoformat(),
              'year':2024,'counts':{a:len(r['rows']) for a,r in runs.items()}, 'imported_counts':imported,
              'continuation_file_sha256':continuation_sha,'binding_sha256':bound_sha,
              'engine_commit':ENGINE_COMMIT,'driver_commit':DRIVER_COMMIT,
              'canonical_gate':gate,'scorer_sources':source_evidence(),
              'model_identity':baseline['config']['model_identity'],
              'protocol_sha256':data['protocol_sha256'],'paired_config_hash':s.hash_json(baseline['common_config']),
              'guidance':{a:r['config']['guidance'] for a,r in runs.items()},
              'inputs':{a:{k:r[k] for k in ('path','manifest_sha256','samples_sha256')} for a,r in runs.items()},
              'generated_code_executed':False,'full_480_per_arm_verified':full,
              'limitations':['Offline checks do not re-tokenize prompts or decode output token IDs.',
                             'Sampling and prompt reconstruction uncertainties remain; this does not recover undisclosed author settings.']}
    return evidence,runs,data


def score(*args, **kwargs):
    evidence,runs,data=audit(*args,**kwargs,full=True)
    scores={a:s.summarize_arm(run['rows'],data,2024) for a,run in runs.items()}
    evidence.update(kind='figure1b_aime2024_paired_text_scores',
                    arms={a:{k:v for k,v in result.items() if k!='rows'} for a,result in scores.items()},
                    adaptive_vs_baseline=s.compare_pair(scores['baseline'],scores['adaptive']))
    return evidence,scores


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('generation','protocol','data-root','engine-root','driver-root','smoke','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source-commit',required=True)
    p.add_argument('--predecessor',type=Path,help='Optional byte-identical local copy of original inputs')
    p.add_argument('--audit-only',action='store_true')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Fresh evidence output required')
    pins=frozen_tree(ROOT,args.source_commit)
    positional=(args.generation,args.protocol,args.data_root,args.engine_root,args.driver_root,args.smoke)
    if args.audit_only:
        evidence,_,_=audit(*positional,predecessor=args.predecessor);scores={}
    else:
        evidence,scores=score(*positional,predecessor=args.predecessor)
    if frozen_tree(ROOT,args.source_commit)!=pins:raise ValueError('Scorer changed during run')
    evidence['scorer_commit']=args.source_commit
    args.output.mkdir(parents=True,exist_ok=False)
    for arm,result in scores.items():
        path=args.output/(arm+'.json');path.write_text(json.dumps(result,indent=2)+'\n')
        evidence['arms'][arm]['score_file_sha256']=sha(path)
    (args.output/'summary.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps({'status':'PASS','audit_only':args.audit_only,'counts':evidence['counts'],'output':str(args.output)}))


if __name__=='__main__':main()
