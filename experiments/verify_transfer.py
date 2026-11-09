"""Skalare Gegenrechnung, historische Abgleiche und deterministische Wiederholung von Versuch 10."""
import argparse
import csv
import json
import tempfile
from pathlib import Path
import numpy as np
from transfer_study import ROOT,OUT,main,sha
from mechanistic_ties import keys


def verify(out):
    if (out/'verification.json').exists():raise FileExistsError('Prüfartefakt existiert bereits')
    provenance=json.loads((out/'provenance.json').read_text())
    previous=json.loads((out/'previous_artifacts.json').read_text())
    for path,digest in {**previous,**provenance['inputs_sha256']}.items():assert sha(ROOT/path)==digest,path
    a=dict(np.load(out/'predictions.npz'))
    old=dict(np.load(ROOT/'reports/mechanistic_ties/mechanism.npz'))
    rows=list(csv.DictReader((out/'results.csv').open()))
    cal=list(csv.DictReader((out/'calibration.csv').open()))
    test_keys=keys('test',a['test_indices'])
    checked=0
    for row in rows:
        name=f"{row['arm']}_s{row['seed']}";prefix=f"{name}_{row['condition']}"
        c=next(v for v in cal if v['arm']==row['arm'] and v['seed']==row['seed'])
        cp=a[name+'_calibration_probabilities']; cc=np.maximum(cp,1-cp)
        order=sorted(zip(cc.tolist(),keys('val',a['calibration_indices']).tolist()),key=lambda v:(-v[0],v[1]))
        boundary,boundary_hash=order[4*len(cc)//5-1]
        assert boundary==float(c['rank_threshold']) and boundary_hash==c['rank_hash']
        tau=float(np.quantile(cc,.2,method='linear')); assert tau==float(c['threshold'])
        correct=zeros=ties=tied_zero=accepted=errors=0; mask=[]
        for p,z,zero,y,key in zip(a[prefix+'_probabilities'],a[prefix+'_logits'],a[prefix+'_zero'],a['labels'],test_keys):
            confidence=max(p,np.float32(1)-p)
            is_correct=(p>=.5)==y
            if zero:assert float(z)==float(c['bias'])
            tie=confidence==tau
            take=confidence>=tau if row['policy']=='standard' else (confidence>boundary or (confidence==boundary and key<=boundary_hash)) and not zero
            correct+=int(is_correct);zeros+=int(zero);ties+=int(tie);tied_zero+=int(tie and zero)
            accepted+=int(take);errors+=int(take and not is_correct);mask.append(take)
        n=len(a['labels'])
        expected=dict(cases=n,correct=correct,accuracy=correct/n,zero_count=zeros,zero_fraction=zeros/n,
            threshold_tie_count=ties,threshold_tie_fraction=ties/n,ties_with_zero=tied_zero,
            accepted=accepted,errors=errors,coverage=accepted/n,risk=errors/accepted if accepted else None)
        for field,value in expected.items():
            if value is None:assert row[field]==''
            else:np.testing.assert_allclose(float(row[field]),value,atol=1e-15,rtol=0)
        np.testing.assert_array_equal(mask,a[prefix+'_'+row['policy']])
        if row['condition'] in ['clean','noise']:
            policy='original' if row['policy']=='standard' else row['policy']
            np.testing.assert_array_equal(mask,old[prefix+'_'+policy])
            np.testing.assert_array_equal(a[prefix+'_zero'],old[prefix+'_zero'])
            checked+=1
    with tempfile.TemporaryDirectory(prefix='transfer-repeat-') as tmp:
        repeat=Path(tmp)/'result';main(repeat)
        b=dict(np.load(repeat/'predictions.npz'))
        assert a.keys()==b.keys()
        for key in a:np.testing.assert_array_equal(a[key],b[key])
        for filename in ['results.csv','calibration.csv','provenance.json','mechanism.png','mechanism.pdf','selection.png','selection.pdf']:
            assert (out/filename).read_bytes()==(repeat/filename).read_bytes(),filename
    for path,digest in previous.items():assert sha(ROOT/path)==digest,path
    result=dict(passed=True,previous_artifacts_unchanged=len(previous),scalar_rows_verified=len(rows),
                study09_decision_rows_verified=checked,deterministic_arrays_verified=len(a),
                deterministic_figures_verified=4,models=6,calibration_cases=262,test_cases=621,
                verifier_sha256=sha(Path(__file__)),outputs_sha256={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='outputs_sha256'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUT)
    verify(parser.parse_args().output)
