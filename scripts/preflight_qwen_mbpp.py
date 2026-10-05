"""CPU-only task entry checks: native controlled decode + all378 real prompts."""
import argparse
import io
import json
import os
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from qwen_mbpp_protocol import *
from generate_humaneval import atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--source-commit',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU only')
    pins=frozen_tree(ROOT,a.source_commit);a.output.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',source_commit=a.source_commit,source_sha256=pins,tests_sha256=sha(ROOT/'tests/test_qwen_mbpp.py'))
    try:
        import torch
        stream=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromName('test_qwen_mbpp')
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        (a.output/'tests.log').write_text(stream.getvalue());print(stream.getvalue(),flush=True)
        report.update(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped))
        if not result.wasSuccessful() or result.testsRun!=6 or result.skipped:raise ValueError('Task CPU tests failed')
        from transformers import AutoTokenizer
        from evalplus.sanitize import sanitize
        tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True,trust_remote_code=False)
        data=load_data(a.project/'data/mbpp/MbppPlus-v0.2.0.jsonl')
        rows=[build_prompt(data['problems'][t],tokenizer) for t in data['task_ids']]
        old=a.project/'results/runs/20261004-qwen-resource-audit'
        audit=json.loads((old/'summary.json').read_text());same(sha(old/'prompts.jsonl'),audit['prompts_sha256'],'frozen resource prompts')
        expected=[json.loads(l) for l in (old/'prompts.jsonl').read_text().splitlines()]
        same([dict(task_id=r['task_id'],prompt=r['prompt'],input_ids=r['prompt_token_ids'],prompt_sha256=hashlib.sha256(r['prompt'].encode()).hexdigest()) for r in rows],expected,'all378 real prompts')
        atomic_json(a.output/'prompts.json',rows)
        # All 378 rows through both-arm validator, synthetic EOS only, never a benchmark.
        source=dict(git_commit=a.source_commit,source_sha256=pins)
        for arm in ARMS:
            cfg=config(arm,source,data['task_ids'],None);samples=[]
            for prompt in rows:
                problem=data['problems'][prompt['task_id']];tokens=[151645];decoded=tokenizer.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                stop=stop_metadata(tokens,decoded,CAP)
                samples.append(dict(**prompt,arm=arm,sample_id=0,generated_token_ids=tokens,generated_tokens=1,
                    raw_generation=tokenizer.decode(tokens,skip_special_tokens=False,clean_up_tokenization_spaces=False),**stop,
                    solution=sanitize(stop['completion'],entrypoint=problem['entry_point']),elapsed_seconds=1.,config_hash=digest(cfg),
                    execution_observation=dict(forward_calls=1,cache_slots=81 if arm=='fixed' else 64,final_cache_length=len(prompt['prompt_token_ids']),
                        layer_calls_per_forward=dict(prelude=15,core=32,strong_tail=17,reference_tail=17 if arm=='fixed' else 0),fresh_cache_initial_length=0)))
            validate_rows(samples,cfg,data,tokenizer,sanitize)
        same(frozen_tree(ROOT,a.source_commit),pins,'final source');report.update(status='PASS',prompt_count=378,synthetic_rows_validated=756,runtime=runtime(),cuda_initialized=torch.cuda.is_initialized(),checkpoint_loaded=False)
        if report['cuda_initialized']:raise ValueError('Unexpected GPU initialization')
    except BaseException as exc:report.update(status='FAIL',error=repr(exc));raise
    finally:
        if (a.output/'tests.log').exists():report['log_sha256']=sha(a.output/'tests.log')
        atomic_json(a.output/'result.json',report)
if __name__=='__main__':main()
