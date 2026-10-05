"""Record native Qwen CPU correctness checks, with exact source/runtime hashes."""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
import unittest
from datetime import datetime, timezone
from continue_aime_figure1b import frozen_tree

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-commit',required=True)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only environment required')
    pins=frozen_tree(ROOT,a.source_commit)
    sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
    import torch
    import transformers.models.qwen3.modeling_qwen3 as native
    import transformers.cache_utils as cache
    import transformers.masking_utils as masking
    import transformers.integrations.sdpa_attention as sdpa
    import test_qwen_loop
    names=unittest.defaultTestLoader.getTestCaseNames(test_qwen_loop.QwenLoopTests)
    if len(names)!=9:raise ValueError('Expected all nine CPU correctness groups')
    a.output.mkdir(parents=True,exist_ok=False)
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(test_qwen_loop.QwenLoopTests)
    with (a.output/'tests.log').open('x') as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    passed=result.wasSuccessful() and result.testsRun==9 and not result.skipped
    source_unchanged=frozen_tree(ROOT,a.source_commit)==pins
    report={'status':'PASS' if passed and source_unchanged else 'FAIL',
        'finished_at':datetime.now(timezone.utc).isoformat(),'kind':'native_random_narrow_cpu_correctness_only',
        'checkpoint_loaded':False,'cuda_initialized':torch.cuda.is_initialized(),
        'source_commit':a.source_commit,'source_sha256':pins,
        'tests_source_sha256':sha(ROOT/'tests/test_qwen_loop.py'),
        'runtime_sha256':{m.__name__:sha(inspect.getfile(m)) for m in (native,cache,masking,sdpa)},
        'versions':{name:importlib.metadata.version(name) for name in ('torch','transformers')},
        'tests':names,'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'skipped':len(result.skipped),'log_sha256':sha(a.output/'tests.log'),
        'interpretation':'No pretrained-weight forward, GPU gate, MBPP generation, or score is represented.'}
    if report['cuda_initialized']:report['status']='FAIL'
    (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','tests_run','errors','failures')}))
    if report['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
