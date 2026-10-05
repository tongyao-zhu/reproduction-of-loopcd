"""Small adapter demonstration; not the HumanEval/AIME benchmark protocol."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--prompt',required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--max-new-tokens',type=int,default=128)
    a=p.parse_args()
    if a.max_new_tokens<1:p.error('max-new-tokens must be positive')
    import torch
    from transformers import GenerationConfig
    from loopcd_repro.runtime import load_ouro
    from loopcd_repro.guidance import GuidanceConfig
    from loopcd_repro.ouro import OuroGuidance
    from generate_aime import new_cache
    model,tok=load_ouro(a.model,a.device)
    prompt=tok.apply_chat_template([{'role':'user','content':a.prompt}],tokenize=False,add_generation_prompt=True)
    ids=torch.tensor([tok.encode(prompt,add_special_tokens=False)],device=a.device)
    for mode in ('baseline','adaptive'):
        torch.manual_seed(42)
        cfg=GenerationConfig(do_sample=False,max_new_tokens=a.max_new_tokens,
            eos_token_id=tok.eos_token_id,pad_token_id=tok.eos_token_id,
            temperature=None,top_p=None,top_k=None,use_cache=True)
        with OuroGuidance(model,GuidanceConfig(mode=mode,omega_cap=1.),total_loops=4),torch.inference_mode():
            out=model.generate(input_ids=ids,generation_config=cfg,past_key_values=new_cache(model))
        print(f'[{mode}]\n'+tok.decode(out[0,ids.shape[1]:],skip_special_tokens=True)+'\n')

if __name__=='__main__':main()
