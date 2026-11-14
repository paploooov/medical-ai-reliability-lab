"""No real HTTP requests or external medical files: fake responses and DICOMs only."""
import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import download_vindr_test as d
import run_vindr_external as v


class Response(io.BytesIO):
    def __init__(self, data, status=200, **headers):
        super().__init__(data)
        self.status = status
        self.headers = {'Content-Length': str(len(data)), **headers}


class UsabilityTests(unittest.TestCase):
    def test_selftest_cli(self):
        proc = subprocess.run([sys.executable, str(v.ROOT/'scripts/run_vindr_external.py'), '--selftest'], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn('Selbsttest bestanden', proc.stdout)
        self.assertEqual(proc.stderr, '')

    def test_missing_installation_and_bad_arguments_are_private(self):
        for args, code in [(['-S', str(v.ROOT/'scripts/run_vindr_external.py'), '--selftest'], 'E1'),
                           ([str(v.ROOT/'scripts/run_vindr_external.py'), '--PRIVATE_SENTINEL'], 'E8')]:
            proc = subprocess.run([sys.executable, *args], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 1)
            self.assertTrue(proc.stderr.startswith(code+':'))
            self.assertNotIn('PRIVATE', proc.stdout+proc.stderr)
            self.assertNotIn('Traceback', proc.stderr)

    def test_error_codes_and_suppression(self):
        for code in v.ERRORS:
            stderr = io.StringIO()
            with patch.object(sys, 'argv', ['run', '--selftest']), patch.object(v, 'check_installation'), patch.object(v, 'selftest', side_effect=v.EvaluationError(code)), contextlib.redirect_stderr(stderr):
                self.assertEqual(v.main(), 1)
            self.assertEqual(stderr.getvalue(), f'{code}: {v.ERRORS[code]}\n')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'labels.csv'
            path.write_text('PRIVATE_SENTINEL\n1\n')
            with self.assertRaises(v.EvaluationError) as caught:
                v.read_labels(path)
            self.assertEqual(caught.exception.code, 'E2')
            path.write_text('image_id,No finding,Pneumonia,Brocho-pneumonia,Pleuro-pneumonia\nFAKE,1,0,0,0\n')
            with self.assertRaises(v.EvaluationError) as caught:
                v.read_labels(path)
            self.assertEqual(caught.exception.code, 'E3')
            with self.assertRaises(v.EvaluationError) as caught:
                v.dicom_to_uint8(object())
            self.assertEqual(caught.exception.code, 'E4')
            with patch.object(v, 'digest', return_value='invalid'), self.assertRaises(v.EvaluationError) as caught:
                v.load_models()
            self.assertEqual(caught.exception.code, 'E5')
            with self.assertRaises(v.EvaluationError) as caught:
                v.run(Path(directory), path)  # output already exists, before reading data
            self.assertEqual(caught.exception.code, 'E6')

    def test_download_scope_and_redirects(self):
        parser = d.TestLinks()
        parser.feed('<a href="fake.dicom">ok</a><a href="../train/private.dicom">bad</a><a href="https://evil.example/test/x.dcm">bad</a><a href="../annotations_test.csv">bad</a>')
        self.assertEqual(parser.names, {'fake.dicom'})
        client = d.Client('FAKE_USER', 'SECRET_PASSWORD')
        with self.assertRaises(d.DownloadError):
            client.open('https://evil.example')
        with self.assertRaises(d.DownloadError):
            d.RestrictedRedirect().redirect_request(Request(d.BASE+'test/'), None, 302, '', {}, 'https://evil.example')
        self.assertFalse(d.allowed(d.BASE+'train/'))
        self.assertFalse(d.allowed(d.BASE+'test/../train/fake.dcm'))

    def test_download_resume_restart_and_completed(self):
        for status in (200, 206):
            with tempfile.TemporaryDirectory() as directory:
                target = Path(directory)/'fake.dicom'
                part = Path(str(target)+'.part')
                part.write_bytes(b'abc')
                response = Response(b'abcdef' if status == 200 else b'def', status,
                                    **({'Content-Range': 'bytes 3-5/6'} if status == 206 else {}))
                client = Mock()
                client.open.return_value = response
                d.download_file(client, d.BASE+'test/fake.dicom', target)
                self.assertEqual(target.read_bytes(), b'abcdef')
                self.assertFalse(part.exists())
                client.open.assert_called_once_with(d.BASE+'test/fake.dicom', 3)
                d.download_file(client, d.BASE+'test/fake.dicom', target)
                self.assertEqual(client.open.call_count, 1)

    def test_download_partial_failure_and_range_416(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'fake.dicom'
            client = Mock()
            client.open.return_value = Response(b'abc', **{'Content-Length':'6'})
            with self.assertRaises(d.DownloadError):
                d.download_file(client, d.BASE+'test/fake.dicom', target)
            self.assertFalse(target.exists())
            part = Path(str(target)+'.part')
            self.assertEqual(part.read_bytes(), b'abc')
            client.open.side_effect = HTTPError('PRIVATE', 416, 'PRIVATE', {'Content-Range':'bytes */3'}, None)
            d.download_file(client, d.BASE+'test/fake.dicom', target)
            self.assertEqual(target.read_bytes(), b'abc')

    def test_login_page_and_mismatched_range_rejected(self):
        for response in (Response(b'PRIVATE', **{'Content-Type':'text/html'}),
                         Response(b'PRIVATE', 206, **{'Content-Range':'bytes 9-15/16'})):
            with tempfile.TemporaryDirectory() as directory:
                target = Path(directory)/'fake.dicom'
                client = Mock()
                client.open.return_value = response
                with self.assertRaises(d.DownloadError):
                    d.download_file(client, d.BASE+'test/fake.dicom', target)
                self.assertFalse(target.exists())

    def test_only_selected_subset_requested(self):
        links = ''.join(f'<a href="fake{i}.dicom">x</a>' for i in range(1397)).encode()
        client = Mock()
        client.open.return_value = Response(links)
        with tempfile.TemporaryDirectory() as directory, patch.object(d, 'download_file') as fetch, contextlib.redirect_stdout(io.StringIO()):
            d.download(client, Path(directory))
            self.assertEqual(fetch.call_count, 1398)
            urls = [call.args[1] for call in fetch.call_args_list]
            self.assertEqual(urls[0], d.BASE+'image_labels_test.csv')
            self.assertTrue(all(url.startswith(d.BASE+'test/') for url in urls[1:]))

    def test_password_and_failure_not_logged_or_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            stderr = io.StringIO()
            with patch.object(sys, 'argv', ['download', '--data-dir', directory]), patch('builtins.input', return_value='FAKE_USER'), patch.object(d.getpass, 'getpass', return_value='SECRET_PASSWORD'), patch.object(d, 'download', side_effect=RuntimeError('SECRET_PASSWORD PRIVATE_ID')), contextlib.redirect_stderr(stderr):
                self.assertEqual(d.main(), 1)
            self.assertTrue(stderr.getvalue().startswith('D1:'))
            self.assertNotIn('SECRET', stderr.getvalue())
            self.assertNotIn('PRIVATE', stderr.getvalue())
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
