"""Investigate failed BF16 cache equivalence, without changing acceptance gates.

Compares native one-pass controls, full-prefix independent loop oracle, and
FP32 computation on the same checkpoint weights. Not a benchmark or new gate.
"""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from continue_aime_figure1b import frozen_tree
from gpu_smoke_qwen_loop import sha,REVISION
from launch_huginn_r16_suite import gpu_status
from generate_humaneval import atomic_json


def oracle(model,ids,cfg):
    import torch
    from transformers.masking_utils import create_causal_mask
    net=model.model;x=net.embed_tokens(ids);pos=torch.arange(ids.shape[1],device=ids.device)
    mask=create_causal_mask(config=net.config,input_embeds=x,attention_mask=None,cache_position=pos,past_key_values=None,position_ids=pos[None])
    rope=net.rotary_emb(x,pos[None])
    def layers(indices,value):
        for i in indices:
            value=net.layers[i](value,attention_mask=mask,position_ids=pos[None],past_key_value=None,use_cache=False,cache_position=pos,position_embeddings=rope)
        return value
    x=layers(range(15),x);states=[x]
    for _ in range(8):x=.875*x+.125*layers(range(15,19),x);states.append(x)
    z=[]
    for state in (states[8],states[1]):z.append(model.lm_head(net.norm(layers(range(19,36),state)[:,-1:])).float())
    return z[0]+.3*(z[0]-z[1])


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--source-commit',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='2' or not gpu_status('2')['ready']:raise ValueError('GPU2 unavailable')
    pins=frozen_tree(ROOT,a.source_commit);a.output.mkdir(parents=True,exist_ok=False)
    failed=a.project/'results/runs/20261004-qwen-gpu-gate/result.json';original=json.loads(failed.read_text())
    if original['status']!='FAIL' or original['checks'][-1]['name']!='incremental/5':raise ValueError('Unexpected failure source')
    report=dict(status='running',source_commit=a.source_commit,source_sha256=pins,failed_gate_sha256=sha(failed),comparisons={},diagnostic_only=True)
    def save():atomic_json(a.output/'result.json',report)
    save()
    try:
        import torch
        from transformers import AutoModelForCausalLM
        from loopcd_repro.qwen_loop import QwenLoopConfig,forward_loop
        snapshot=Path('/path/to/your/workspace/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots')/REVISION
        for name,pin in original['model_files'].items():
            if sha(snapshot/name)!=pin['sha256']:raise ValueError('Changed model '+name)
        if not gpu_status('2')['ready']:raise ValueError('GPU2 became busy')
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model,info=AutoModelForCausalLM.from_pretrained(snapshot,torch_dtype=torch.bfloat16,attn_implementation='sdpa',
            device_map={'':'cuda:0'},local_files_only=True,trust_remote_code=False,output_loading_info=True)
        if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):raise ValueError('Non-strict model')
        model.eval();ids=torch.tensor([[151643,100,200,300,400,500,600]],device='cuda:0');cfg=QwenLoopConfig()
        def compare(name,a,b):
            a=a.float();b=b.float();delta=(a-b).abs();threshold=.06+.02*b.abs()
            report['comparisons'][name]=dict(max_abs_error=float(delta.max()),mean_abs_error=float(delta.mean()),
                outside_original_tolerance=int((delta>threshold).sum()),total=delta.numel(),
                argmax_a=a.argmax(-1).tolist(),argmax_b=b.argmax(-1).tolist(),finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
                tight_fp32_allclose=bool(torch.allclose(a,b,atol=.0001,rtol=.0001)))
            save()
        with torch.inference_mode():
            for dtype in (torch.bfloat16,torch.float32):
                label=str(dtype);model.to(dtype=dtype)
                native_full=model(ids,use_cache=False).logits[:,4:5]
                native_prefix=model(ids[:,:4],use_cache=True)
                native_increment=model(ids[:,4:5],past_key_values=native_prefix.past_key_values,use_cache=True).logits
                compare(label+'/native_cache',native_increment,native_full)
                del native_full,native_prefix,native_increment
                full=forward_loop(model,ids,cfg,use_cache=False,last_token_only=False)
                prefix5=forward_loop(model,ids[:,:5],cfg,use_cache=False)
                compare(label+'/future_mask',prefix5.logits,full.logits[:,4:5])
                independent=oracle(model,ids[:,:5],cfg)
                compare(label+'/independent_oracle',prefix5.logits,independent)
                prefix=forward_loop(model,ids[:,:4],cfg)
                incremental=forward_loop(model,ids[:,4:5],cfg,cache=prefix.cache)
                compare(label+'/guided_cache',incremental.logits,full.logits[:,4:5])
                compare(label+'/strong_cache',incremental.strong_logits,full.strong_logits[:,4:5])
                compare(label+'/weak_cache',incremental.reference_logits,full.reference_logits[:,4:5])
                compare(label+'/cache_vs_oracle',incremental.logits,independent)
                tail=forward_loop(model,ids[:,5:],cfg,cache=prefix.cache,last_token_only=False)
                compare(label+'/two_token_cache',tail.logits,full.logits[:,5:])
                del full,prefix5,independent,prefix,incremental,tail
                torch.cuda.empty_cache()
        if frozen_tree(ROOT,a.source_commit)!=pins:raise ValueError('Source changed')
        report.update(status='completed',max_memory_allocated=torch.cuda.max_memory_allocated());save()
    except BaseException as e:report.update(status='failed',error=repr(e));save();raise
if __name__=='__main__':main()
