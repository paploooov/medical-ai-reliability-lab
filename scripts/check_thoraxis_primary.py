"""Independent main-number check from saved predictions; no inference, no analysis imports."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/original_512/thoraxis_30'

def main():
    manifest=json.loads((OUT/'prediction_manifest.json').read_text());risks={}
    plan={r['run_id']:r for r in json.loads((ROOT/'configs/thoraxis_training_batch.json').read_text())['runs']}
    assert manifest['completed'] and len(manifest['models'])==30
    for item in manifest['models']:
        path=OUT/item['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
        with np.load(path,allow_pickle=False) as z:
            m=z['partition']=='test';labels=z['label'][m];p=expit(z['logit'][m]);c=np.maximum(p,1-p);errors=(p>=.5)!=labels
            assert len(labels)==618 and int(sum(labels==0))==231
            expected_rows={r['name']:r for r in json.loads((ROOT/plan[item['run_id']]['manifest']).read_text())['records'] if r['split']=='test'}
            assert set(z['image_id'][m])==set(expected_rows)
            for name,label,group,origin in zip(z['image_id'][m],labels,z['group'][m],z['origin'][m]):
                expected=expected_rows[str(name)]
                assert (int(label),str(group),str(origin))==(expected['label'],expected['group'],expected['original_split'])
            assert len(set(z['image_id'][m]))==618
            assert not set(z['group'][m])&set(z['group'][~m])
        # Independent implementation: walk tied blocks in descending confidence.
        rows=sorted(zip(c.tolist(),errors.tolist()),key=lambda x:-x[0]);remaining=.8*len(rows);bad=0.;i=0
        while remaining>1e-12:
            j=i+1
            while j<len(rows) and rows[j][0]==rows[i][0]:j+=1
            used=min(remaining,j-i);bad+=used*sum(error for _,error in rows[i:j])/(j-i)
            remaining-=used;i=j
        risks[item['run_id']]=bad/(.8*len(rows))
    contrasts=[[risks[f'official_seed{s}']-risks[f'{split}_seed{s}'] for s in [17,42,83,123,2026]] for split in range(20261112,20261117)]
    primary=sum(map(sum,contrasts))/25
    expected=json.loads((OUT/'h1_results.json').read_text());assert abs(primary-expected['primary_mean'])<1e-12
    table=list(csv.DictReader((OUT/'h1_contrasts.csv').open()))
    for i,row in enumerate(table):
        for j,s in enumerate([17,42,83,123,2026]):assert abs(float(row[f'training_seed_{s}'])-contrasts[i][j])<1e-12
    result=dict(passed=True,independent_primary_mean=primary,absolute_difference=abs(primary-expected['primary_mean']),models_checked=30,contrasts_checked=25,
                method='Separate sorted-block implementation; frozen predictions only; all saved file hashes verified')
    (OUT/'independent_check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
