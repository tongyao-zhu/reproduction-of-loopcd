"""Synthetic full 480x2 end-to-end scoring and adversarial lineage checks."""
import copy
import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch

sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'scripts'),str(Path(__file__).resolve().parent)]
import score_aime_figure1b as f
from test_score_aime import FullGenerationTests


class FigureScoring(unittest.TestCase):
    def setUp(self):
        self.base=FullGenerationTests('test_complete_three_arm_end_to_end')
        self.base.setUp();self.addCleanup(self.base.doCleanups)
        self.root=self.base.output
        self.old=self.root/'original';self.old.mkdir()
        self.smoke=self.root/'smoke.json';self.smoke.write_text('{}')
        self.driver_pins={'scripts/continue_aime_figure1b.py':'d'*64}
        proofs={}
        for arm in ('baseline','fixed','adaptive'):
            p=self.root/arm
            m=json.loads((p/'manifest.json').read_text())
            m['config']['source']['git_commit']=f.ENGINE_COMMIT
            self.engine_pins=m['config']['source']['source_sha256']
            m['config_hash']=f.s.hash_json(m['config'])
            m['paired_config_hash']=f.s.hash_json({k:v for k,v in m['config'].items() if k!='guidance'})
            m['gpu_smoke_sha256']=f.sha(self.smoke)
            rows=[json.loads(l) for l in (p/'samples.jsonl').read_text().splitlines()]
            for row in rows:
                row['config_hash']=m['config_hash'];row['paired_config_hash']=m['paired_config_hash']
            self.base.save(p,m,rows)
            op=self.old/arm;op.mkdir()
            before=copy.deepcopy(m);before.update(status='running',completed_samples=4)
            self.base.save(op,before,rows[:4])
            proofs[arm]={'n':4,'manifest_sha256':f.sha(op/'manifest.json'),'samples_sha256':f.sha(op/'samples.jsonl'),'config_hash':m['config_hash']}
        self.binding=dict(schema_version=1,kind='figure1b_aime2024_continuation',engine_commit=f.ENGINE_COMMIT,
            engine_source_sha256=self.engine_pins,driver_commit=f.DRIVER_COMMIT,driver_source_sha256=self.driver_pins,
            arms=list(f.ARMS),year=2024,samples_per_arm=480,imported_sample_bytes_unchanged=True,
            generation_parameters_unchanged=True,predecessor=str(self.old),predecessor_files=proofs)
        (self.root/'continuation.json').write_text(json.dumps({'binding':self.binding}))
        self.bound=f.s.hash_json(self.binding)
        for arm in f.ARMS:
            p=self.root/arm/'manifest.json';m=json.loads(p.read_text())
            m['execution_driver']={'commit':f.DRIVER_COMMIT,'continuation_binding_sha256':self.bound}
            p.write_text(json.dumps(m))
        (self.root/'status.json').write_text(json.dumps({'status':'completed_generation_unscored',
             'binding_sha256':self.bound,'completed':{a:480 for a in f.ARMS}}))
        def tree(root,commit):
            return self.engine_pins if commit==f.ENGINE_COMMIT else self.driver_pins
        self.addCleanup(patch.stopall)
        patch.object(f,'frozen_tree',side_effect=tree).start()
        patch.object(f,'validate_smoke').start()  # No model/GPU claims in these synthetic tests.
        self.args=(self.root,self.base.protocol,self.base.data_root,self.root/'engine',self.root/'driver',self.smoke)

    def test_full_pair_scored_and_canonical60(self):
        result,scores=f.score(*self.args)
        self.assertEqual(result['counts'],{'baseline':480,'adaptive':480})
        self.assertEqual(result['canonical_gate']['canonical_tasks'],60)
        self.assertEqual(result['canonical_gate']['boundary_cases_passed'],25)
        self.assertEqual(result['arms']['adaptive']['correct_samples'],480)
        self.assertEqual(result['adaptive_vs_baseline']['sample_ties'],480)
        self.assertFalse(result['generated_code_executed'])

    def test_missing_row_refused_as_full(self):
        path=self.root/'adaptive'
        m=json.loads((path/'manifest.json').read_text());rows=[json.loads(l) for l in (path/'samples.jsonl').read_text().splitlines()][:-1]
        m.update(status='running',completed_samples=479);self.base.save(path,m,rows)
        with self.assertRaises(ValueError):f.score(*self.args)

    def test_partial_audit_never_reports_score(self):
        for arm in f.ARMS:
            path=self.root/arm;m=json.loads((path/'manifest.json').read_text())
            rows=[json.loads(l) for l in (path/'samples.jsonl').read_text().splitlines()][:4]
            m.update(status='running',completed_samples=4);self.base.save(path,m,rows)
        result,_,_=f.audit(*self.args)
        self.assertTrue(result['audit_only']);self.assertFalse(result['full_480_per_arm_verified'])
        self.assertNotIn('arms',result);self.assertNotIn('adaptive_vs_baseline',result)

    def test_predecessor_tamper(self):
        p=self.old/'fixed/samples.jsonl';p.write_text(p.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'unchanged old samples'):f.score(*self.args)

    def test_wrong_driver_binding(self):
        p=self.root/'adaptive/manifest.json';m=json.loads(p.read_text());m['execution_driver']['commit']='a'*40;p.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,'driver manifest'):f.score(*self.args)

    def test_changed_imported_whitespace_refused(self):
        p=self.root/'baseline/samples.jsonl';lines=p.read_text().splitlines();lines[0]=' '+lines[0]
        p.write_text('\n'.join(lines)+'\n')
        mp=p.parent/'manifest.json';m=json.loads(mp.read_text());m['samples_sha256']=f.sha(p);mp.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,'Imported row bytes'):f.score(*self.args)

    def test_changed_token_pair_refused(self):
        path=self.root/'adaptive';m=json.loads((path/'manifest.json').read_text())
        rows=[json.loads(l) for l in (path/'samples.jsonl').read_text().splitlines()]
        for row in rows[16:32]:
            row['prompt_token_ids'][1]=101;row['prompt_token_ids_sha256']=f.s.hash_json(row['prompt_token_ids'])
        self.base.save(path,m,rows)
        with self.assertRaisesRegex(ValueError,'paired row'):f.score(*self.args)

    def test_duplicate_id_refused(self):
        path=self.root/'adaptive';m=json.loads((path/'manifest.json').read_text())
        rows=[json.loads(l) for l in (path/'samples.jsonl').read_text().splitlines()];rows[-1]=rows[-2]
        self.base.save(path,m,rows)
        with self.assertRaises(ValueError):f.score(*self.args)

    def test_completed_driver_required(self):
        p=self.root/'status.json';v=json.loads(p.read_text());v['status']='running';p.write_text(json.dumps(v))
        with self.assertRaisesRegex(ValueError,'completed driver'):f.score(*self.args)


if __name__=='__main__':unittest.main()
