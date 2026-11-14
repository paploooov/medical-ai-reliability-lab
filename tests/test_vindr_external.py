"""Synthetic-only tests. Never read a clinical image or label file."""
import csv
import io
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset, Dataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_vindr_external as v


def fake_dicom(path, bits=12, mono='MONOCHROME2', window=False, inverse_pixels=False):
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b'\0'*128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.PatientName = 'PRIVATE_SYNTHETIC_SENTINEL'
    ds.Rows, ds.Columns = 96, 80
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = mono
    ds.BitsAllocated = 8 if bits == 8 else 16
    ds.BitsStored, ds.HighBit, ds.PixelRepresentation = bits, bits-1, 0
    values = np.rint(np.linspace(0, 2**bits-1, ds.Rows*ds.Columns)).reshape(ds.Rows, ds.Columns)
    if inverse_pixels:
        values = 2**bits-1-values
    ds.PixelData = values.astype(np.uint8 if bits == 8 else '<u2').tobytes()
    if window:
        ds.WindowCenter = [2**(bits-1), 1]
        ds.WindowWidth = [2**bits, 2]
        ds.RescaleSlope, ds.RescaleIntercept = 1, 0
    ds.save_as(path, enforce_file_format=True)
    return ds


def fake_labels(path, n=8):
    fields = ['image_id', 'rad_id', 'No finding', 'Pneumonia', 'Brocho-pneumonia', 'Pleuro-pneumonia', 'Bronchitis']
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i in range(n):
            row = {k: 0 for k in fields}
            row.update(image_id=f'PRIVATE_ID_{i}', rad_id='PRIVATE_READER')
            row[['No finding', 'Pneumonia', 'Brocho-pneumonia', 'Bronchitis'][i % 4]] = 1
            w.writerow(row)


class SyntheticTests(unittest.TestCase):
    def test_bits_inversion_window_and_letterbox(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'fake.dcm'
            for bits in (8, 12, 16):
                for window in (False, True):
                    a = v.dicom_to_uint8(fake_dicom(p, bits, window=window))
                    b = v.dicom_to_uint8(fake_dicom(p, bits, mono='MONOCHROME1', window=window, inverse_pixels=True))
                    np.testing.assert_array_equal(a, b)
                    self.assertEqual(a.dtype, np.uint8)
                    out = v.letterbox(a)
                    self.assertEqual(out.shape, (512, 512))
                    self.assertTrue(np.all(out[:, :42] == 0))
            # Same training letterbox, independently imported from existing code.
            from prepare_original_xray import letterbox
            np.testing.assert_array_equal(v.letterbox(a), letterbox(v.Image.fromarray(a), 512))

    def test_lut_padding_signed_and_invalid(self):
        with tempfile.TemporaryDirectory() as d:
            ds = fake_dicom(Path(d)/'fake.dcm', bits=8, window=True)
            lut = Dataset()
            lut.LUTDescriptor = [256, 0, 8]
            lut.LUTData = list(range(256))
            ds.VOILUTSequence = [lut]
            ds.WindowWidth = 2
            ds.WindowCenter = 128
            a = v.dicom_to_uint8(ds)
            self.assertGreater(len(np.unique(a)), 10)  # LUT preferred to narrow window
            del ds.VOILUTSequence
            ds = fake_dicom(Path(d)/'fake.dcm', bits=16)
            ds.PixelRepresentation = 1
            ds.PixelData = np.linspace(-1000, 1000, 96*80).astype('<i2').tobytes()
            ds.PixelPaddingValue = -1000
            self.assertEqual(int(v.dicom_to_uint8(ds)[0, 0]), 0)
            ds.PhotometricInterpretation = 'RGB'
            with self.assertRaises(ValueError):
                v.dicom_to_uint8(ds)
            ds.PhotometricInterpretation = 'MONOCHROME2'
            ds.PixelData = np.zeros((96, 80), '<i2').tobytes()
            with self.assertRaises(ValueError):
                v.dicom_to_uint8(ds)

    def test_label_mapping_and_duplicates(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'labels.csv'
            fake_labels(p)
            rows, excluded = v.read_labels(p, 8)
            self.assertEqual([r[1] for r in rows], [0,1,1,-1]*2)
            self.assertEqual(excluded, 0)
            original = p.read_text()
            content = original.replace('PRIVATE_ID_1', 'PRIVATE_ID_0')
            p.write_text(content)
            with self.assertRaises(ValueError):
                v.read_labels(p, 8)
            lines = original.splitlines()
            # Conflict and empty vector must be excluded, never silently made normal.
            lines[1] = 'PRIVATE_ID_0,PRIVATE_READER,1,1,0,0,0'
            lines[2] = 'PRIVATE_ID_1,PRIVATE_READER,0,0,0,0,0'
            p.write_text('\n'.join(lines)+'\n')
            rows, excluded = v.read_labels(p, 8)
            self.assertEqual(excluded, 2)
            self.assertEqual([rows[0][1],rows[1][1]], [-2,-2])

    def test_rank_ties_and_class_coverage(self):
        p = np.array([.9, .9, .1, .1, .6], np.float32)
        y = np.array([0,1,0,1,1])
        weights = v.rank_weights(np.maximum(p,1-p), np.ones(5))
        self.assertAlmostEqual(float(weights.sum()), 4)
        np.testing.assert_array_equal(weights[:4], np.ones(4))
        # Independent all-tie expectation: exactly equal .8 weight for each image.
        np.testing.assert_allclose(v.rank_weights(np.ones(5), np.ones(5)), .8)
        rw = v.rank_weights(np.maximum(p,1-p), np.ones(5))
        overall = v.metrics(y,p,.95,np.ones(5),rw=rw)
        self.assertEqual(overall['R80'], .5)
        self.assertTrue(np.isnan(overall['fixed_accepted_error']))
        normal = v.metrics(y,p,.5,np.ones(5),mask=y==0,rw=rw)
        self.assertEqual(normal['R80_coverage'], 1)

    def test_bootstrap_and_privacy(self):
        y = np.array([0,1,0,1,-1])
        logits = np.tile(np.array([-2,2,2,-2,1], np.float32), (30,1))
        result = v.summarize(y, logits, np.full(30,.8), np.array(['PRIVATE_A','PRIVATE_A','PRIVATE_B','PRIVATE_B','PRIVATE_C']), 50)
        serialized = json.dumps(v.clean(result), allow_nan=False)
        self.assertNotIn('PRIVATE', serialized)
        self.assertEqual(result['primary_group_count'], 2)
        self.assertEqual(result['comparisons']['R80']['contrast_mean'], 0)
        self.assertEqual(result['bootstrap']['R80']['contrast']['percentile95'], [0,0])
        self.assertEqual(result['models'][0]['control']['n'], 1)

    def test_safe_cli_failure(self):
        proc = subprocess.run([sys.executable, str(v.ROOT/'scripts/run_vindr_external.py'), '--data-dir', '/PRIVATE_SENTINEL_MISSING', '--out', '/PRIVATE_SENTINEL_OUTPUT'], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 1)
        self.assertNotIn('PRIVATE_SENTINEL', proc.stdout+proc.stderr)
        self.assertNotIn('Traceback', proc.stderr)

    def test_real_checkpoint_synthetic_inference_and_export(self):
        import torch
        torch.set_num_threads(1)
        models, cutoffs, hashes = v.load_models()
        self.assertEqual(len(set(hashes)), 30)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'test').mkdir()
            fake_labels(root/'image_labels_test.csv')
            arrays = []
            for i in range(8):
                ds = fake_dicom(root/'test'/f'PRIVATE_ID_{i}.dicom', bits=(8,12,16)[i%3], mono='MONOCHROME1' if i%2 else 'MONOCHROME2')
                arrays.append(v.letterbox(v.dicom_to_uint8(ds)))
                if i == 0:
                    del ds.StudyInstanceUID
                    ds.save_as(root/'test'/f'PRIVATE_ID_{i}.dicom', enforce_file_format=True)
            start = time.monotonic()
            actual = v.predict(models, arrays[:4])
            elapsed = time.monotonic()-start
            x = torch.tensor(np.stack(arrays[:4]), dtype=torch.float32)[:,None].repeat(1,3,1,1)/255
            x = (x-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None]
            with torch.inference_mode():
                direct = np.stack([m(x).flatten().numpy() for m in models])
            np.testing.assert_array_equal(actual, direct)
            read = v.read_labels
            with patch.object(v, 'read_labels', lambda p: read(p, 8)), patch.object(v, 'load_models', return_value=(models,cutoffs,hashes)):
                v.run(root, root/'result.json')
                with self.assertRaises(ValueError):
                    v.run(root, root/'result.json')
            output = (root/'result.json').read_text()
            self.assertNotIn('PRIVATE', output)
            result = json.loads(output)
            self.assertEqual(len(result['models']), 30)
            self.assertEqual(result['total_label_records'], 8)
            self.assertEqual(result['grouping'], 'published_one_image_per_study_proxy')
            self.assertEqual(result['bootstrap']['R80']['contrast']['valid_replicates'], 2000)
            print(f'SYNTHETIC BENCHMARK: 4 images x 30 models = {elapsed:.3f} seconds; synthetic end-to-end 8 images = {result["total_seconds"]:.3f} seconds')


if __name__ == '__main__':
    unittest.main()
