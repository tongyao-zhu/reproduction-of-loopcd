import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from test_qwen_loop import model_fixture
from qwen_stream_oracle import oracle_stream
from loopcd_repro.qwen_loop import forward_loop,QwenLoopConfig

class StreamOracleTests(unittest.TestCase):
    def test_separate_native_cache_oracle(self):
        torch.set_num_threads(2)
        for dtype in (torch.float32,torch.bfloat16):
            model=model_fixture(dtype);ids=torch.tensor([[1,3,8,2,16,21,6]])
            past=None;cache=None
            with torch.inference_mode():
                for start,end in ((0,4),(4,5),(5,7)):
                    actual=forward_loop(model,ids[:,start:end],QwenLoopConfig(),cache=cache,last_token_only=False);cache=actual.cache
                    expected,past=oracle_stream(model,ids[:,start:end],past)
                    torch.testing.assert_close(actual.logits,expected,atol=0,rtol=0)
                    self.assertEqual(past['length'],cache.length)
if __name__=='__main__':unittest.main()
