"""Check published statistics and README against extracted evidence; no model inference."""
import hashlib
import json
import math
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def display(value, signed=False):
    rounded = Decimal(str(value)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    return format(rounded, '+.2f' if signed else '.2f')

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
            n,b,l = row.get('samples_per_arm',row['n']),row['baseline_correct'],row['loopcd_correct']
            require(type(n) is int and n>0 and all(type(x) is int and 0<=x<=n for x in (b,l)), 'Invalid counts')
            for key,expected in [('baseline_percent',100*b/n),('loopcd_percent',100*l/n),('delta_pp',100*(l-b)/n)]:
                require(math.isclose(row[key],expected,abs_tol=1e-10), 'Count/percentage mismatch')
            path = ROOT/'results'/row['evidence'];raw=path.read_bytes()
            require(hashlib.sha256(raw).hexdigest()==row['evidence_sha256'], 'Evidence changed')
            ev=json.loads(raw);require(ev['id']==row['id'],'Wrong evidence row')
            require(ev.get('n',ev.get('n_tasks',ev.get('n_documents')))==row['n'],'Evidence problem count')
            if row['id'].startswith('aime-'):
                require(row['n']==30 and row['samples_per_problem']==ev['samples_per_problem']==16 and ev['samples_per_arm']==n==480,'AIME coverage')
                require(ev['full_480_per_arm_verified'] and ev['correct_counts']=={'baseline':b,'adaptive':l},'AIME counts')
                comp=ev['comparison'];m=comp['metrics']['pass_at_1']
                problems=comp['problems']
                expected={f'AIME2024-{part}-{idx:02d}' for part in ('I','II') for idx in range(1,16)}
                require(len(problems)==30 and {p['task_id'] for p in problems}==expected,'AIME problem IDs')
                require(all(type(p[k]) is int and 0<=p[k]<=16 for p in problems for k in ('baseline_c_i','candidate_c_i')),'AIME problem counts')
                require(sum(p['baseline_c_i'] for p in problems)==b and sum(p['candidate_c_i'] for p in problems)==l,'AIME problem sums')
                require(all(math.isclose(p['delta_pass_at_1'],(p['candidate_c_i']-p['baseline_c_i'])/16,abs_tol=1e-10) for p in problems),'AIME problem deltas')
                before=m['baseline_percent'];after=m['candidate_percent']
                pair=dict(wins=comp['sample_wins'],losses=comp['sample_losses'],ties=comp['sample_ties'],delta_percentage_points=m['delta_percentage_points'])
            elif row['metric']=='acc_norm':
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
            ours=f"{display(row['baseline_percent'])} → {display(row['loopcd_percent'])}";delta=display(row['delta_pp'],signed=True)
            complete+=1;positive+=l>b
        lines.append(f"| {row['task']} | {row['model']} | {paper} | {ours} | {row['paper_delta']:+.2f} / {delta} |")
    require(complete==data['completed'], 'Completion count')
    readme=(ROOT/'README.md').read_text().split('<!-- RESULTS:START -->')[1].split('<!-- RESULTS:END -->')[0].strip()
    require(readme=='\n'.join(lines),'README table differs from results JSON')
    print(f'PASS: {complete}/8 complete comparisons, {positive} positive; {8-complete} pending. Evidence hashes, counts, paired deltas and README agree.')
    print('This validates published statistics, not a new inference run or raw-output rescore.')

if __name__=='__main__':
    verify()
