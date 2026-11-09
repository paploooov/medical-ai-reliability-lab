"""Independent aggregation and byte-identical figure replay; preserve released artifacts."""
from pathlib import Path
import argparse,csv,json,hashlib,tempfile,statistics
from make_overview import ROOT,build,CONDITIONS

def main(out):
    if (out/'verification.json').exists():raise FileExistsError('Existing verification left untouched')
    rows=list(csv.DictReader((ROOT/'reports/transfer/results.csv').open()))
    summary=json.loads((out/'overview_summary.json').read_text())
    for arm,conditions in summary['metrics_percent_mean_min_max'].items():
        for condition,metrics in conditions.items():
            for metric,expected in metrics.items():
                vals=[100*float(r[metric]) for r in rows if r['arm']==arm and r['condition']==condition and r['policy']=='standard']
                actual=[statistics.mean(vals),min(vals),max(vals)]
                assert all(abs(a-b)<1e-12 for a,b in zip(actual,expected))
    assert summary['panel_b']['calibration_accepted']==236
    assert summary['panel_b']['noise_accepted']==602
    baseline=json.loads((ROOT/'docs/releases/v1.0.0/artifacts.sha256.json').read_text())
    for name,digest in baseline.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    with tempfile.TemporaryDirectory() as tmp:
        other=Path(tmp)/'figures';build(other)
        for name in ['overview_figure.png','overview_figure.pdf','overview_summary.json','overview_provenance.json']:
            assert (out/name).read_bytes()==(other/name).read_bytes(),name
    result=dict(passed=True,released_artifacts_unchanged=len(baseline),metric_summaries_checked=32,figures_byte_identical=2,
                source_hashes_verified=True,verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for name,digest in json.loads((out/'overview_provenance.json').read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'reports/figures')
    main(parser.parse_args().output)
