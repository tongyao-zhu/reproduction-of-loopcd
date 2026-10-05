"""Exercise the actual future GPU numerical suite on a real tiny CPU model."""
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from gpu_smoke_qwen_loop import numerical
from test_qwen_loop import model_fixture


class QwenGPUGateTests(unittest.TestCase):
    def test_all_gpu_check_branches_on_cpu(self):
        torch.set_num_threads(2)
        report={'checks':[]}
        numerical(model_fixture(),report,device='cpu',long_length=65)
        self.assertEqual(len(report['checks']),16)
        self.assertEqual(len({c['name'] for c in report['checks']}),16)
        self.assertTrue(all(c['passed'] for c in report['checks']))
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main()
