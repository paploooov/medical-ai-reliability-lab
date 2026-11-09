"""Alle bisherigen Prüfprogramme in einer isolierten Kopie; Originalartefakte bleiben unverändert."""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from mechanistic_ties import ROOT, OUT, sha


def main(destination):
    destination.mkdir(exist_ok=False)
    commands = [
        ['-m', 'unittest', 'discover', '-s', 'tests', '-v'],
        ['experiments/audit.py'],
        ['experiments/verify_resolution.py'],
        ['experiments/verify_selective_review.py'],
        ['experiments/verify_calibration.py'],
        ['experiments/budget_analysis.py'],
        ['experiments/augmentation_analysis.py'],
        ['scripts/compare_resolutions.py'],
    ]
    results = []
    env = dict(os.environ, OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', PYTHONHASHSEED='0')
    with tempfile.TemporaryDirectory(prefix='medical-ai-checks-') as tmp:
        checkout = Path(tmp) / 'repo'
        shutil.copytree(ROOT, checkout, ignore=shutil.ignore_patterns('.git', '.venv', 'data', '__pycache__', 'project_checks'))
        (checkout / 'data').symlink_to(ROOT / 'data', target_is_directory=True)
        # Some historical analyses record git rev-parse; no Git mutations are run.
        (checkout / '.git').symlink_to(ROOT / '.git', target_is_directory=True)
        for i, args in enumerate(commands):
            run = subprocess.run([sys.executable, *args], cwd=checkout, env=env,
                                 text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            (destination / f'{i:02d}.log').write_text(run.stdout)
            results.append(dict(command='python ' + ' '.join(args), exit_code=run.returncode, log=f'{i:02d}.log'))
            print(results[-1], flush=True)
        for folder in ['audit', 'resolution', 'selective_review', 'calibration', 'budget', 'augmentation']:
            for name in ['verification.json', 'artifact_checks.json', 'resolution_64.json']:
                source = checkout / 'reports' / folder / name
                if source.exists():
                    shutil.copy2(source, destination / f'{folder}_{name}')
    previous = json.loads((OUT / 'previous_artifacts.json').read_text())
    for path, digest in previous.items():
        assert sha(ROOT / path) == digest, path
    result = dict(commands=results, previous_artifacts_unchanged=len(previous),
                  passed=all(r['exit_code'] == 0 for r in results))
    (destination / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    if not result['passed']:
        raise RuntimeError('Mindestens eine Prüfung fehlgeschlagen; siehe Logs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT / 'project_checks')
    main(parser.parse_args().output)
