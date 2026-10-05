import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from download_parcae370 import fetch

class Download(unittest.TestCase):
    def test_exact_bytes_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'blob';data=b'pinned fixture'
            spec=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
            fetch('fixture',p,spec,lambda n:None,opener=lambda *a,**k:io.BytesIO(data))
            self.assertEqual(p.read_bytes(),data)
            with self.assertRaises(FileExistsError):fetch('fixture',p,spec,lambda n:None)
    def test_corrupt_partial_not_promoted(self):
        for data in (b'',b'x'*3,b'x'*10):
            with self.subTest(data=data),tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)/'blob';spec=dict(bytes=3,sha256=hashlib.sha256(b'abc').hexdigest())
                with self.assertRaises(ValueError):fetch('fixture',p,spec,lambda n:None,opener=lambda *a,**k:io.BytesIO(data))
                self.assertFalse(p.exists());self.assertTrue(p.with_suffix('.partial').exists())

if __name__=='__main__':unittest.main()
