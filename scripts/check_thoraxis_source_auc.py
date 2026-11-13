"""Independent pair-comparison AUROC check, no sklearn or analysis imports."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/original_512/thoraxis_30'
def main():
    meta=json.loads((OUT/'source_oof_manifest.json').read_text());path=OUT/meta['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['sha256']
    z=np.load(path,allow_pickle=False);y=z['label'];p=z['score'];positive=p[y==1];negative=p[y==0]
    assert len(positive)>0 and len(negative)>0 and len(set(z['image_id']))==len(y)
    folds=json.loads((ROOT/'configs/thoraxis_source_folds.json').read_text());seen=[]
    assert list(z['image_id'])==[r['name'] for r in folds['records']]
    for f in folds['folds']:
        a,b,c=[set(z['group'][ix]) for ix in [f['train'],f['selection'],f['test']]]
        assert not a&b and not a&c and not b&c
        for ix in f['test']:assert z['fold'][ix]==f['fold']
        seen.extend(f['test'])
    assert sorted(seen)==list(range(len(y)))
    auc=sum(float((s>negative).sum())+.5*float((s==negative).sum()) for s in positive)/(len(positive)*len(negative))
    reported=json.loads((OUT/'source_cnn_results.json').read_text())['auroc'];assert abs(auc-reported)<1e-12
    result=dict(passed=True,pairwise_auroc=auc,reported_auroc=reported,absolute_difference=abs(auc-reported),n=len(y),normal_only=all(r['label']==0 for r in folds['records']),group_isolation_checked=True)
    assert result['normal_only'];(OUT/'source_auc_independent_check.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
