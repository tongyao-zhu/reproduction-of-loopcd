"""Render the eight Figure 1b comparisons from verified published results.

Requires matplotlib. Run `python scripts/verify_results.py` first.
"""
import json
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]

def number(x):
    return str(Decimal(str(x)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))

def main():
    data = json.loads((ROOT / 'results/figure1b.json').read_text())
    rows = data['rows']
    assert len(rows) == 8 and all(r['status'] == 'complete' for r in rows)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.labelcolor': '#333333', 'text.color': '#252525',
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.3), sharey=True)
    fig.subplots_adjust(left=.255, right=.98, top=.79, bottom=.23, wspace=.13)
    blue = '#4c72a8'
    for ax, prefix, title in zip(axes, ['paper', 'ours'], ['Paper · Figure 1b', 'Our reproduction']):
        for i, row in enumerate(rows):
            b = row['paper_baseline'] if prefix == 'paper' else row['baseline_percent']
            l = row['paper_loopcd'] if prefix == 'paper' else row['loopcd_percent']
            delta = row['paper_delta'] if prefix == 'paper' else row['delta_pp']
            assert l >= b
            ax.barh(i, b, height=.56, color='white', edgecolor='#aaaaaa', hatch='////', linewidth=.8, zorder=3)
            ax.barh(i, l-b, left=b, height=.56, color=blue, zorder=3)
            ax.text(l+1.2, i, '+'+number(delta), va='center', fontsize=10.5, fontweight='bold')
        ax.set_xlim(0, 100)
        ax.set_xticks([0,20,40,60,80,100])
        ax.set_xlabel('Score (%)', labelpad=9)
        ax.set_title(title, loc='left', pad=16, fontsize=14, fontweight='bold')
        ax.grid(axis='x', color='#eeeeee', zorder=0)
        ax.tick_params(axis='both', length=0, pad=8)
        for s in ('top','right'): ax.spines[s].set_visible(False)
        for s in ('bottom','left'): ax.spines[s].set_color('#bbbbbb')
    axes[0].set_yticks(range(8), [r['task']+'  |  '+r['model']+('*' if r['id']=='qwen-mbpp' else '') for r in rows])
    axes[0].invert_yaxis()
    fig.suptitle('LoopCD at the same depth', x=.04, y=.98, ha='left', fontsize=21, fontweight='bold')
    fig.text(.04,.925,'Eight completed comparisons · gain labels are percentage points',fontsize=12,color='#555555')
    fig.legend(handles=[Patch(facecolor='white',edgecolor='#aaaaaa',hatch='////',label='Baseline'),
                        Patch(facecolor=blue,label='Gain with LoopCD')],loc='upper right',bbox_to_anchor=(.98,.94),frameon=False,ncol=2)
    fig.text(.04,.105,'AIME: 30 problems × 16 samples per arm. Our gain 95% problem-bootstrap intervals: 2.6B [4.79, 13.54]; 1.4B [0.83, 10.83].',fontsize=9.5)
    fig.text(.04,.073,'HumanEval / MBPP: base-test pass@1. ARC-C / HellaSwag: length-normalized accuracy. Other uncertainty is in the result extracts.',fontsize=9.5)
    fig.text(.04,.041,'* Looped-Qwen3 uses our independent reconstruction. Protocol differences remain; matching the direction does not mean matching the paper.',fontsize=9.5)
    fig.text(.04,.01,'Source: arXiv:2610.02185v1 and results/figure1b.json · '+data['updated'],fontsize=8.5,color='#666666')
    for ext in ('png','svg','pdf'):
        fig.savefig(ROOT/'figures'/('figure1b-comparison.'+ext),dpi=180,facecolor='white')
    plt.close(fig)

if __name__ == '__main__': main()
