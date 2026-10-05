"""Pinned Parcae-370M ARC-C 25-shot full input audit, without model inference."""
import argparse
from pathlib import Path
from audit_parcae_prompts import audit, ROOT

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'harness-reference', 'hub-cache', 'dataset-cache', 'output'):
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--registry', default=ROOT / 'configs/mc_datasets.json', type=Path)
    audit(p.parse_args(), profile='parcae370_arc_challenge_v1')

if __name__ == '__main__':
    main()
