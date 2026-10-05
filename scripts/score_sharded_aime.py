"""Validate full disjoint shard union and score only complete paired AIME."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from loopcd_repro.aime_protocol import load_protocol,load_questions,build_prompt,read_model_identity,sha256_file,atomic_json
from loopcd_repro.aime_vllm_protocol import require,scoring_rows
from sharded_aime_common import common,verify_source,read_batches,load_predecessor,expected_keys,combine_unique

def score(project,model,runs,output):
    from transformers import AutoTokenizer
    from score_aime import load_scoring_data,canonical_gate,summarize_arm,compare_pair
    require(not output.exists(),'Output already exists')
    protocol=load_protocol(project/'data/aime/protocol-v1.json')
    data=load_scoring_data(project/'data/aime/protocol-v1.json',project/'data/aime/prepared-v1')
    gate=canonical_gate(data)
    tok=AutoTokenizer.from_pretrained(model,trust_remote_code=True,local_files_only=True)
    prompts=[build_prompt(tok,q,protocol) for q in load_questions(project/'data/aime/prepared-v1',2024)]
    identity=read_model_identity(model,protocol)
    manifests=[json.loads((run/'manifest.json').read_text()) for run in runs]
    require(len(runs)==2 and common(manifests[0])==common(manifests[1]),'Unpaired configuration')
    require([(m['shard']['task_start'],m['shard']['task_end']) for m in manifests]==[(0,18),(18,30)],'Unexpected shard partition')
    all_parts=[];evidence=[]
    for run,m in zip(runs,manifests):
        require(json.loads((run/'status.json').read_text())['status']=='completed_generation','Shard incomplete')
        verify_source(project,m)
        require(m['prompts']==prompts and m['model_identity']==identity,'Input/model changed')
        require(m['adaptive_cap']==protocol['models'][identity['repo_id']]['adaptive_cap'],'Cap changed')
        require(m['protocol_sha256']==sha256_file(project/'data/aime/protocol-v1.json') and
                m['questions_sha256']==sha256_file(project/'data/aime/prepared-v1/aime2024.questions.jsonl'),'Data binding')
        require(json.loads((run/'canonical/canonical_gate.json').read_text())==gate and
                sha256_file(run/'canonical/canonical_gate.json')==m['canonical_sha256'],'Canonical binding')
        binding=m['shard']['predecessor']
        previous,bound=load_predecessor(project,Path(binding['path']) if binding else None,m,tok)
        require(bound==binding,'Lineage binding changed')
        new,hashes=read_batches(run,m,tok)
        expected=expected_keys(prompts,m['shard']['task_start'],m['shard']['task_end'])
        combine_unique([previous,new],expected)
        all_parts.extend([previous,new])
        evidence.append(dict(run=str(run),manifest_sha256=sha256_file(run/'manifest.json'),batch_sha256=hashes,predecessor=bound))
    union=combine_unique(all_parts,expected_keys(prompts))
    scores={arm:summarize_arm(scoring_rows([union[(arm,p['task_id'],i)] for p in prompts for i in range(0,16,2)]),data,2024) for arm in ('baseline','adaptive')}
    result=dict(status='PASS',full_480_per_arm_verified=True,canonical_gate=gate,sources=evidence,scores=scores,
                comparison=compare_pair(scores['baseline'],scores['adaptive']),
                limitation='Same vLLM batch2 numerical code; explicitly recorded process/GPU/task sharding and immutable predecessor lineage. No HF samples.')
    output.mkdir(parents=True,exist_ok=False);atomic_json(output/'comparison.json',result)
    print(json.dumps(result['comparison']['metrics']))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('project','model','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--runs',type=Path,nargs=2,required=True)
    a=p.parse_args();score(a.project,a.model,a.runs,a.output)
