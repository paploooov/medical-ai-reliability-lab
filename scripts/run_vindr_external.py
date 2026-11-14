"""Offline VinDr-PCXR evaluation. Never serialize patient-level data or exceptions."""
import argparse
import contextlib
import os
import csv
import hashlib
import json
import logging
import re
import sys
import time
import warnings
import functools
import importlib
import tempfile
from pathlib import Path

_IMPORT_FAILED = False
try:
    import numpy as np
    import PIL
    from PIL import Image
    from scipy.special import expit
    from scipy.stats import t
except Exception:
    _IMPORT_FAILED = True

ERRORS = {
    'E1': 'Abhängigkeit fehlt oder Installation ist nicht funktionsfähig.',
    'E2': 'Labeldatei fehlt oder Labelschema ist ungültig.',
    'E3': 'Bildanzahl oder Zuordnung zwischen Bildern und Labels stimmt nicht.',
    'E4': 'DICOM-Format oder synthetische Pipeline-Prüfung fehlgeschlagen.',
    'E5': 'Checkpoint oder Manifest fehlt, ist ungültig oder hat einen falschen Hash.',
    'E6': 'Ausgabepfad nicht verfügbar oder Ergebnisdatei existiert bereits.',
    'E7': 'Berechnung fehlgeschlagen; keine Datenwerte ausgegeben.',
    'E8': 'Aufruf prüfen: entweder --selftest oder --data-dir und --out.',
}


class EvaluationError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


@contextlib.contextmanager
def stage(code):
    try:
        yield
    except EvaluationError:
        raise
    except Exception:
        raise EvaluationError(code) from None


def coded(code):
    def decorate(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            with stage(code):
                return func(*args, **kwargs)
        return wrapped
    return decorate


@coded('E1')
def check_installation():
    require(not _IMPORT_FAILED)
    for name in ('torch', 'torchvision', 'pydicom', 'numpy', 'scipy', 'PIL',
                 'pylibjpeg', 'libjpeg', 'openjpeg', 'jpeg_ls'):
        importlib.import_module(name)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'configs/thoraxis_vindr_external.json'
SEEDS = [17, 42, 83, 123, 2026]
SPLITS = list(range(20261112, 20261117))
RUNS = [f'official_seed{s}' for s in SEEDS] + [f'{sp}_seed{s}' for sp in SPLITS for s in SEEDS]
POSITIVE = {'pneumonia', 'brochopneumonia', 'pleuropneumonia'}


def require(condition):
    if not condition:
        raise ValueError('Invalid input')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(name):
    name = re.sub('[^a-z0-9]', '', name.lower())
    return 'brochopneumonia' if name == 'bronchopneumonia' else name


@coded('E2')
def read_labels(path, expected=1397):
    """Only named one-hot columns; fail closed on ambiguous vectors or duplicates."""
    records, excluded = [], 0
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        normalized = [canonical(k) for k in fields]
        require(len(set(normalized)) == len(fields))
        columns = dict(zip(normalized, fields))
        require({'imageid', 'nofinding'} | POSITIVE <= columns.keys())
        labels = {k: v for k, v in columns.items() if k not in {'imageid', 'radid'}}
        seen = set()
        for row in reader:
            require(None not in row)
            image_id = row[columns['imageid']]
            require(bool(re.fullmatch(r'[A-Za-z0-9_-]+', image_id)))
            require(image_id not in seen)
            seen.add(image_id)
            values = {k: float(row[v]) for k, v in labels.items()}
            require(all(v in (0, 1) for v in values.values()))
            active = {k for k, v in values.items() if v == 1}
            if not active or ('nofinding' in active and len(active) > 1):
                excluded += 1
                label = -2
            elif active == {'nofinding'}:
                label = 0
            elif active & POSITIVE:
                label = 1
            else:
                label = -1  # descriptive control, NOT a negative pneumonia label
            records.append((image_id, label))
    if len(records) != expected:
        raise EvaluationError('E3')
    return records, excluded


@coded('E4')
def dicom_to_uint8(ds):
    from pydicom.pixels import apply_modality_lut, apply_voi_lut
    require(ds.PhotometricInterpretation in ('MONOCHROME1', 'MONOCHROME2'))
    require(int(getattr(ds, 'NumberOfFrames', 1)) == 1 and int(ds.SamplesPerPixel) == 1)
    raw = ds.pixel_array
    require(raw.ndim == 2)
    valid = np.ones(raw.shape, bool)
    if 'PixelPaddingValue' in ds:
        low, high = sorted([float(ds.PixelPaddingValue), float(getattr(ds, 'PixelPaddingRangeLimit', ds.PixelPaddingValue))])
        valid &= (raw < low) | (raw > high)
    values = np.asarray(apply_voi_lut(apply_modality_lut(raw, ds), ds, index=0, prefer_lut=True), dtype=np.float64)
    require(valid.any() and np.isfinite(values[valid]).all())
    low, high = values[valid].min(), values[valid].max()
    require(high > low)
    values = np.clip((values - low) / (high - low), 0, 1)
    if ds.PhotometricInterpretation == 'MONOCHROME1':
        values = 1 - values
    values[~valid] = 0
    return np.rint(values * 255).astype(np.uint8)


def letterbox(array):
    image = Image.fromarray(array).convert('L')
    w, h = image.size
    scale = 512 / max(w, h)
    resized = image.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)
    out = Image.new('L', (512, 512), 0)
    out.paste(resized, ((512 - resized.width) // 2, (512 - resized.height) // 2))
    return np.asarray(out)


@coded('E5')
def load_models():
    import torch
    from torchvision.models import mobilenet_v3_small
    protocol = json.loads(PROTOCOL.read_text())
    manifest_path = ROOT / protocol['checkpoint_manifest']
    require(digest(manifest_path) == protocol['checkpoint_manifest_sha256'])
    manifest = json.loads(manifest_path.read_text())
    entries = manifest['checkpoints']
    require(len(entries) == 30 and {e['run_id'] for e in entries} == set(RUNS))
    models, cutoffs, hashes = [], [], []
    by_name = {e['run_id']: e for e in entries}
    for name in RUNS:
        entry = by_name[name]
        path = manifest_path.parent / entry['file']
        require(digest(path) == entry['sha256'])
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        require(checkpoint['run_id'] == name and checkpoint['input_size'] == 512)
        model = mobilenet_v3_small(weights=None)
        model.classifier = torch.nn.Linear(576, 1)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model.eval()
        cutoff = float(checkpoint['cutoff'])
        require(.5 <= cutoff <= 1)
        models.append(model)
        cutoffs.append(cutoff)
        hashes.append(entry['sha256'])
    reference = models[0].features[:10].state_dict()
    for model in models[1:]:
        other = model.features[:10].state_dict()
        require(reference.keys() == other.keys())
        require(all(torch.equal(v, other[k]) for k, v in reference.items()))
    return models, np.array(cutoffs), hashes


def predict(models, arrays):
    import torch
    x = torch.tensor(np.stack(arrays), dtype=torch.float32)[:, None].repeat(1, 3, 1, 1) / 255
    x = (x - torch.tensor([.485, .456, .406])[None, :, None, None]) / torch.tensor([.229, .224, .225])[None, :, None, None]
    with torch.inference_mode():
        shared = models[0].features[:10](x)
        logits = [m.classifier(torch.flatten(m.avgpool(m.features[10:](shared)), 1)).flatten().numpy() for m in models]
    result = np.stack(logits)
    require(np.isfinite(result).all())
    return result


def rank_weights(confidence, mass):
    levels, inverse = np.unique(confidence, return_inverse=True)
    totals = np.bincount(inverse, weights=mass, minlength=len(levels))
    higher = np.cumsum(totals[::-1])[::-1] - totals
    frac = np.divide(np.clip(.8 * mass.sum() - higher, 0, totals), totals,
                     out=np.zeros_like(totals), where=totals > 0)
    return mass * frac[inverse]


def ratio(a, b):
    return float(a / b) if b > 0 else float('nan')


def metrics(y, prob, cutoff, mass, mask=None, rw=None):
    confidence = np.maximum(prob, 1 - prob)
    pred = prob >= .5
    error = pred != y
    accepted = confidence >= cutoff
    if rw is None:
        rw = rank_weights(confidence, mass)
    if mask is None:
        mask = np.ones(len(y), bool)
    w, rankw = mass * mask, rw * mask
    aw = w * accepted
    return {
        'n': float(w.sum()), 'R80': ratio(rankw @ error, rankw.sum()),
        'R80_coverage': ratio(rankw.sum(), w.sum()),
        'specificity': ratio(w @ ((y == 0) & ~pred), w @ (y == 0)),
        'sensitivity': ratio(w @ ((y == 1) & pred), w @ (y == 1)),
        'fixed_acceptance': ratio(aw.sum(), w.sum()),
        'fixed_accepted_error': ratio(aw @ error, aw.sum()),
        'fixed_accepted_count': float(aw.sum()), 'fixed_accepted_errors': float(aw @ error),
        'error': ratio(w @ error, w.sum()),
    }


def variation(values):
    values = np.asarray(values, float)
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    half = float(t.ppf(.975, 4) * sd / np.sqrt(5))
    return dict(values=values.tolist(), mean=mean, sd=sd, range=[float(values.min()), float(values.max())], t95=[mean-half, mean+half], df=4)


def contrast(values):
    values = np.asarray(values)
    return values[:5][None, :] - values[5:].reshape(5, 5)


def interval(values):
    values = np.asarray(values)
    valid = values[np.isfinite(values)]
    return dict(valid_replicates=len(valid), percentile95=np.quantile(valid, [.025, .975]).tolist() if len(valid) else None)


def summarize(y, logits, cutoffs, groups, replicates=2000):
    """All identifiers are consumed in RAM; returned object contains aggregates only."""
    keep, control = y >= 0, y == -1
    require(np.any(y == 0) and np.any(y == 1))
    probabilities = expit(logits)  # keep float32 behavior of internal evaluation
    yy, pp = y[keep], probabilities[:, keep]
    _, gi = np.unique(groups[keep], return_inverse=True)
    ng = int(gi.max() + 1)
    require(ng >= 2)
    mass = np.ones(len(yy))
    rows = []
    for i, name in enumerate(RUNS):
        confidence = np.maximum(pp[i], 1-pp[i])
        rw = rank_weights(confidence, mass)
        overall = metrics(yy, pp[i], cutoffs[i], mass, rw=rw)
        accepted = confidence >= cutoffs[i]
        accepted_groups = len(np.unique(gi[accepted]))
        zero_upper = 1 - .05 ** (1 / accepted_groups) if accepted_groups and overall['fixed_accepted_errors'] == 0 else None
        cp = probabilities[i, control]
        cc = np.maximum(cp, 1-cp)
        rows.append(dict(run_id=name, cutoff=float(cutoffs[i]), overall=overall,
                         normal=metrics(yy, pp[i], cutoffs[i], mass, yy == 0, rw),
                         pneumonia=metrics(yy, pp[i], cutoffs[i], mass, yy == 1, rw),
                         accepted_groups=accepted_groups, zero_error_group_risk_upper95=zero_upper,
                         control=dict(n=int(control.sum()), positive_fraction=float(np.mean(cp >= .5)) if len(cp) else None,
                                      fixed_acceptance=float(np.mean(cc >= cutoffs[i])) if len(cp) else None,
                                      mean_confidence=float(cc.mean()) if len(cp) else None)))
    keys = ['R80', 'specificity', 'sensitivity', 'fixed_acceptance', 'fixed_accepted_error']
    point = {}
    for key in keys:
        values = np.array([r['overall'][key] for r in rows])
        table = contrast(values)
        point[key] = dict(official_mean=float(values[:5].mean()), mixed_mean=float(values[5:].mean()),
                          official_seed_variation=variation(values[:5]),
                          mixed_seed_variation=variation(values[5:].reshape(5, 5).mean(axis=0)),
                          mixed_split_variation=variation(values[5:].reshape(5, 5).mean(axis=1)),
                          contrast_mean=float(table.mean()), contrast_5x5=table.tolist(),
                          contrast_seed_variation=variation(table.mean(axis=0)),
                          contrast_split_variation=variation(table.mean(axis=1)))
    boot = {key: {arm: [] for arm in ('official', 'mixed', 'contrast')} for key in keys}
    rng = np.random.default_rng(20261117)
    for _ in range(replicates):
        gm = np.bincount(rng.integers(0, ng, ng), minlength=ng)
        mass = gm[gi].astype(float)
        sample = [metrics(yy, pp[i], cutoffs[i], mass) for i in range(30)]
        for key in keys:
            values = np.array([r[key] for r in sample])
            official, mixed = float(values[:5].mean()), float(values[5:].mean())
            boot[key]['official'].append(official)
            boot[key]['mixed'].append(mixed)
            boot[key]['contrast'].append(official-mixed)
    return dict(primary_group_count=ng, models=rows, comparisons=point,
                bootstrap={key: {arm: interval(v) for arm, v in arms.items()} for key, arms in boot.items()})


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run(data_dir, out):
    import pydicom
    import torch
    import torchvision
    import scipy
    with stage('E6'):
        require(not out.exists() and out.parent.is_dir())  # never overwrite results
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    start = time.monotonic()
    models, cutoffs, hashes = load_models()
    protocol = json.loads(PROTOCOL.read_text())
    records, excluded = read_labels(data_dir / protocol['label_file'])
    with stage('E3'):
        files = {p.stem: p for p in (data_dir / 'test').glob('*.dicom')}
        dcm = list((data_dir / 'test').glob('*.dcm'))
        require(not (files.keys() & {p.stem for p in dcm}))
        files.update({p.stem: p for p in dcm})
        require(set(files) == {r[0] for r in records})
    arrays, predictions, groups, labels = [], [], [], []
    for image_id, label in records:
        if label == -2:
            continue
        with stage('E4'):
            ds = pydicom.dcmread(files[image_id])
            groups.append(str(getattr(ds, 'StudyInstanceUID', '')).strip())
            labels.append(label)
            arrays.append(letterbox(dicom_to_uint8(ds)))
        if len(arrays) == 4:
            predictions.append(predict(models, arrays))
            arrays.clear()
    if arrays:
        predictions.append(predict(models, arrays))
    # Public documentation removes patient identifiers; do not infer identity from them.
    complete_study_uids = all(groups)
    if not complete_study_uids:
        groups = [str(i) for i in range(len(groups))]
    inference_seconds = time.monotonic() - start
    result = summarize(np.array(labels), np.concatenate(predictions, axis=1), cutoffs, np.array(groups))
    result.update(dict(dataset='VinDr-PCXR 1.0.0 official test', protocol_sha256=digest(PROTOCOL),
                       checkpoint_sha256=dict(zip(RUNS, hashes)), code_sha256=digest(Path(__file__)),
                       training_seeds=SEEDS, split_seeds=SPLITS, bootstrap_replicates=2000, bootstrap_seed=20261117,
                       grouping='StudyInstanceUID' if complete_study_uids else 'published_one_image_per_study_proxy',
                       patient_linkage_available=False, excluded_label_records=excluded,
                       total_label_records=len(records), inference_seconds=inference_seconds,
                       total_seconds=time.monotonic()-start,
                       versions=dict(torch=torch.__version__, torchvision=torchvision.__version__, numpy=np.__version__,
                                     scipy=scipy.__version__, pydicom=pydicom.__version__, Pillow=PIL.__version__),
                       warnings=['Study bootstrap cannot guarantee patient independence.',
                                 'Zero observed errors or a degenerate bootstrap interval do not imply zero risk.',
                                 'Radiological labels are not clinical confirmation.']))
    # Exclusive creation prevents overwriting results even when concurrent runs finish.
    with stage('E6'):
        with out.open('x', encoding='utf-8') as f:
            json.dump(clean(result), f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')


def selftest():
    """Installation and real checkpoint checks, but only generated image pixels."""
    check_installation()
    import torch
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    models, _, _ = load_models()
    arrays = []
    with stage('E4'), tempfile.TemporaryDirectory() as directory:
        for bits in (8, 12, 16):
            for window in (False, True):
                pair = []
                for inverted in (False, True):
                    meta = FileMetaDataset()
                    meta.TransferSyntaxUID = ExplicitVRLittleEndian
                    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
                    meta.MediaStorageSOPInstanceUID = generate_uid()
                    path = Path(directory) / 'synthetic.dcm'
                    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b'\0' * 128)
                    ds.Rows, ds.Columns, ds.SamplesPerPixel = 32, 24, 1
                    ds.PhotometricInterpretation = 'MONOCHROME1' if inverted else 'MONOCHROME2'
                    ds.BitsAllocated = 8 if bits == 8 else 16
                    ds.BitsStored, ds.HighBit, ds.PixelRepresentation = bits, bits - 1, 0
                    pixels = np.rint(np.linspace(0, 2**bits-1, 32*24)).reshape(32, 24)
                    if inverted:
                        pixels = 2**bits-1-pixels
                    ds.PixelData = pixels.astype(np.uint8 if bits == 8 else '<u2').tobytes()
                    if window:
                        ds.WindowCenter, ds.WindowWidth = 2**(bits-1), 2**bits
                    ds.save_as(path, enforce_file_format=True)
                    converted = dicom_to_uint8(pydicom.dcmread(path))
                    require(converted.dtype == np.uint8 and converted.min() == 0 and converted.max() == 255)
                    pair.append(converted)
                require(np.array_equal(*pair))
                image = letterbox(pair[0])
                require(image.shape == (512, 512) and not image[:, :64].any())
                arrays.append(image)
    with stage('E1'):
        output = predict(models, arrays[:2])
        require(output.shape == (30, 2) and np.isfinite(output).all())


class SafeParser(argparse.ArgumentParser):
    def error(self, message):
        raise EvaluationError('E8')


def main():
    warnings.filterwarnings('ignore')
    logging.disable(logging.CRITICAL)
    try:
        parser = SafeParser(description='Offline-Auswertung oder synthetischer Selbsttest.')
        parser.add_argument('--data-dir', type=Path)
        parser.add_argument('--out', type=Path)
        parser.add_argument('--selftest', action='store_true')
        args = parser.parse_args()
        if (args.selftest and (args.data_dir or args.out)) or (not args.selftest and not (args.data_dir and args.out)):
            raise EvaluationError('E8')
        with open(os.devnull, 'w') as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            check_installation()
            if args.selftest:
                selftest()
            else:
                run(args.data_dir, args.out)
    except Exception as error:
        code = error.code if isinstance(error, EvaluationError) else 'E7'
        print(f'{code}: {ERRORS[code]}', file=sys.stderr)
        return 1
    print('Selbsttest bestanden: Installation, 30 Checkpoints und synthetische DICOM-Pipeline.' if args.selftest
          else 'Fertig. Nur die aggregierte Ergebnis-JSON weitergeben.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
