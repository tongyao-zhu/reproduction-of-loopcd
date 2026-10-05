"""Check published statistics and README against extracted evidence; no model inference."""
import hashlib
import json
import math
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def require(condition, message):
    if not condition:
        raise ValueError(message)

def verify():
    data = json.loads((ROOT/'results/figure1b.json').read_text())
    require(len(data['rows']) == 8 and len({r['id'] for r in data['rows']}) == 8, 'Expected eight distinct rows')
    lines = ['| Benchmark | Model | Paper baseline → LoopCD | Ours baseline → LoopCD | Δ paper / ours |',
             '|---|---|---:|---:|---:|']
    complete = positive = 0
    for row in data['rows']:
        require(math.isclose(row['paper_loopcd']-row['paper_baseline'],row['paper_delta'],abs_tol=1e-10), 'Paper delta')
        paper = f"{row['paper_baseline']:.2f} → {row['paper_loopcd']:.2f}"
        if row['status'] == 'in_progress':
            require(row['id'] in ('aime-14','aime-26'), 'Unexpected pending benchmark')
            require(all(row[k] is None for k in ('baseline_correct','loopcd_correct','baseline_percent','loopcd_percent','delta_pp')), 'Pending row has a score')
            ours = 'Running'; delta = '—'
        else:
            require(row['status'] == 'complete', 'Unknown result state')
            n,b,l = row['n'],row['baseline_correct'],row['loopcd_correct']
            require(type(n) is int and n>0 and all(type(x) is int and 0<=x<=n for x in (b,l)), 'Invalid counts')
            for key,expected in [('baseline_percent',100*b/n),('loopcd_percent',100*l/n),('delta_pp',100*(l-b)/n)]:
                require(math.isclose(row[key],expected,abs_tol=1e-10), 'Count/percentage mismatch')
            path = ROOT/'results'/row['evidence'];raw=path.read_bytes()
            require(hashlib.sha256(raw).hexdigest()==row['evidence_sha256'], 'Evidence changed')
            ev=json.loads(raw);require(ev['id']==row['id'],'Wrong evidence row')
            require(ev.get('n',ev.get('n_tasks',ev.get('n_documents')))==n,'Evidence sample count')
            if row['metric']=='acc_norm':
                m=ev['metrics']['acc_norm'];pair=m['paired_vs_baseline8']['adaptive8']
                require(m['correct_counts']=={'baseline8':b,'adaptive8':l},'Evidence counts')
                before=m['percent']['baseline8'];after=m['percent']['adaptive8']
            else:
                m=pair=ev['metrics']['base'];before=m['baseline_pass_at_1_percent']
                key=next(k for k in ('adaptive_pass_at_1_percent','hidden_pass_at_1_percent','fixed_pass_at_1_percent') if k in m)
                after=m[key]
            require(math.isclose(before,row['baseline_percent'],abs_tol=1e-10) and math.isclose(after,row['loopcd_percent'],abs_tol=1e-10), 'Evidence percentages')
            require(pair['wins']+pair['losses']+pair['ties']==n and pair['wins']-pair['losses']==l-b,'Paired outcomes')
            require(math.isclose(pair['delta_percentage_points'],row['delta_pp'],abs_tol=1e-10),'Paired delta')
            ours=f"{row['baseline_percent']:.2f} → {row['loopcd_percent']:.2f}";delta=f"{row['delta_pp']:+.2f}"
            complete+=1;positive+=l>b
        lines.append(f"| {row['task']} | {row['model']} | {paper} | {ours} | {row['paper_delta']:+.2f} / {delta} |")
    require(complete==data['completed']==6, 'Completion count')
    readme=(ROOT/'README.md').read_text().split('<!-- RESULTS:START -->')[1].split('<!-- RESULTS:END -->')[0].strip()
    require(readme=='\n'.join(lines),'README table differs from results JSON')
    print(f'PASS: {complete}/8 complete comparisons, {positive} positive, 1 zero; 2 pending. Evidence hashes, counts, paired deltas and README agree.')
    print('This validates published statistics, not a new inference run or raw-output rescore.')

if __name__=='__main__':
    verify()
