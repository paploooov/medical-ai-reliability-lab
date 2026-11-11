import hashlib,io,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from reproduce_512 import download_archive

class DownloadTests(unittest.TestCase):
    def test_identified_request_and_atomic_verified_archive(self):
        payload=b'known archive bytes'
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'archive.zip'
            with patch('reproduce_512.urllib.request.urlopen',return_value=io.BytesIO(payload)) as call:
                download_archive('https://example.org/data',target,hashlib.sha256(payload).hexdigest())
            self.assertEqual(call.call_args.args[0].get_header('User-agent'),'THORAXIS-reproduction/1.0')
            self.assertEqual(target.read_bytes(),payload)
            self.assertFalse(target.with_suffix('.partial').exists())
    def test_invalid_download_never_becomes_final_archive(self):
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'archive.zip'
            with patch('reproduce_512.urllib.request.urlopen',return_value=io.BytesIO(b'wrong')):
                with self.assertRaisesRegex(ValueError,'SHA-256'):download_archive('https://example.org/data',target,'0'*64)
            self.assertFalse(target.exists())
            self.assertEqual(target.with_suffix('.partial').read_bytes(),b'wrong')
