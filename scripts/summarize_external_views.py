"""Summarize saved AP/PA results without images, training or threshold changes."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ('accuracy', 'sensitivity', 'specificity', 'coverage', 'retained_error')


def summarize_views(sample, results):
    rows = []
    for view in ('AP', 'PA'):
        cases = [r for r in sample if r['arm'] == 'primary' and r['view'] == view]
        positives = sum(int(r['label']) for r in cases)
        for arm in ('control_28', 'augmented_28', 'control_64'):
            for condition in ('clean', 'noise', 'relative_blur', 'contrast'):
                group = [r for r in results if r['dataset'] == 'external_' + view
                         and r['arm'] == arm and r['condition'] == condition]
                if len(group) != 3 or {int(r['seed']) for r in group} != {17, 42, 83}:
                    raise ValueError(f'Missing or duplicate seeds: {view}/{arm}/{condition}')
                if any(int(r['n']) != len(cases) for r in group):
                    raise ValueError('Sample and results counts disagree')
                row = dict(view=view, arm=arm, condition=condition, n=len(cases),
                           positive=positives, negative=len(cases)-positives,
                           prevalence=positives / len(cases))
                row.update({key: sum(float(r[key]) for r in group) / 3 for key in METRICS})
                row['balanced_accuracy'] = (row['sensitivity'] + row['specificity']) / 2
                rows.append(row)
    return rows


def main():
    with (ROOT / 'configs/external_rsna_sample.csv').open() as f:
        sample = list(csv.DictReader(f))
    with (ROOT / 'reports/external/results.csv').open() as f:
        results = list(csv.DictReader(f))
    rows = summarize_views(sample, results)
    with (ROOT / 'reports/external/view_summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        if row['condition'] == 'clean':
            print(row)


if __name__ == '__main__':
    main()
