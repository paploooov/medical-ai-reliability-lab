"""Synthetic row-isolation checks and exact protocol schedule; no training."""
import json
from pathlib import Path
import sys
import unittest
import tempfile
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from train_thoraxis_protocol import development_rows
import run_thoraxis_batch as batch

class TrainingBatch(unittest.TestCase):
    def test_cache_alignment_uses_ids_and_excludes_own_test(self):
        source=[dict(name=f'image{i}',group=f'g{i}',label=i%2,split='train') for i in range(4)]
        assigned=[dict(source[i],split=part) for i,part in [(3,'test'),(2,'selection'),(0,'train'),(1,'calibration')]]
        records,ids=development_rows(source,assigned)
        self.assertEqual(ids.tolist(),[2,0,1])
        self.assertNotIn('image3',[r['name'] for r in records])
        sentinel=np.arange(4)*10
        self.assertEqual(sentinel[ids].tolist(),[20,0,10])

    def test_duplicate_or_changed_provenance_is_rejected(self):
        source=[dict(name='a',group='ga',label=0,split='train'),dict(name='b',group='gb',label=1,split='test')]
        for assigned in ([source[0],source[0]],[dict(source[0],label=1),source[1]]):
            with self.assertRaises(ValueError):development_rows(source,assigned)

    def test_exact_crossed_schedule(self):
        p=json.loads((ROOT/'configs/thoraxis_training_batch.json').read_text())
        expected={(str(split),seed) for split in ['official',*range(20261112,20261117)] for seed in [17,42,83,123,2026]}
        self.assertEqual(len(p['runs']),30)
        self.assertEqual({(r['split'],r['seed']) for r in p['runs']},expected)
        self.assertEqual(len({r['run_id'] for r in p['runs']}),30)

class BackgroundRecovery(unittest.TestCase):
    def test_failure_retries_same_seed_and_keeps_both_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);local=root/'local';local.mkdir()
            plan={'runs':[{'run_id':'official_seed17','split':'official','seed':17,'manifest':'source.json'}],
                  'max_attempts_per_run':3,'timeout_seconds':60,'input_sha256':{'source.json':'sourcehash'}}
            planpath=root/'plan.json';planpath.write_text(json.dumps(plan));calls=[]
            def fake_train(command,**kwargs):
                calls.append(command)
                if len(calls)==1:return SimpleNamespace(returncode=1)
                out=Path(command[command.index('--output')+1]);out.mkdir()
                checkpoint=out/'checkpoint.pt';checkpoint.write_bytes(b'synthetic checkpoint, no model')
                result={'seed':17,'run_id':'official_seed17','test_evaluated':False,'medical_test_predictions':0,
                        'checkpoint_sha256':batch.sha(checkpoint),'manifest_sha256':'sourcehash',
                        'selected_epoch':1,'selected_selection_nll':0.25,'total_seconds':0.1}
                (out/'training_run.json').write_text(json.dumps(result))
                return SimpleNamespace(returncode=0)
            with patch.multiple(batch,ROOT=root,LOCAL=local,PLAN=planpath,STATE=local/'state.json',
                                SUMMARY=root/'summary.csv',LOGS=root/'logs.json',REPORT=root/'report.md'), \
                 patch.object(batch,'verify_plan'),patch.object(batch,'commit_outputs'), \
                 patch.object(batch.subprocess,'run',side_effect=fake_train):
                batch.main()
            state=json.loads((local/'state.json').read_text())
            self.assertEqual(state['status'],'completed')
            attempts=state['runs']['official_seed17']['attempts']
            self.assertEqual([a['status'] for a in attempts],['failed','completed'])
            self.assertIn('Exit code 1',attempts[0]['reason'])
            self.assertEqual([c[c.index('--seed')+1] for c in calls],['17','17'])

if __name__=='__main__':unittest.main()
