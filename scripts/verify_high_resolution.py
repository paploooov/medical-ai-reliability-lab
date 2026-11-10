"""Independently verify saved high-resolution probes, source hashes and metrics."""
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score,accuracy_score,confusion_matrix,brier_score_loss
ROOT=Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/'reports/high_resolution';summary=json.loads((folder/'summary.json').read_text())
    data=np.load(folder/'predictions_and_probes.npz');cfg=summary['config'];provenance=summary['provenance']
    assert hashlib.sha256((ROOT/'data/pneumoniamnist_224.npz').read_bytes()).hexdigest()==provenance['archive_sha256']
    assert hashlib.sha256((ROOT/'data/torch/hub/checkpoints/resnet18-f37072fd.pth').read_bytes()).hexdigest()==provenance['encoder_sha256']
    for name,digest in provenance['feature_sha256'].items():
        assert hashlib.sha256((ROOT/'data/high_resolution_features'/name).read_bytes()).hexdigest()==digest
    for view in cfg['views']:
        result=summary['results'][view]
        candidates=[c for c in summary['selection_candidates'] if c['view']==view]
        assert result['C']==min(candidates,key=lambda c:c['selection_nll'])['C']
        for split in ['calibration','test']:
            features=np.load(ROOT/f'data/high_resolution_features/{view}_{split}.npz')
            np.testing.assert_array_equal(features['labels'],data[split+'_labels'])
            z=((features['features']-data[view+'_scaler_mean'])/data[view+'_scaler_scale'])@data[view+'_coefficient'].ravel()+data[view+'_intercept'][0]
            np.testing.assert_allclose(z,data[view+'_'+split+'_logits'],atol=1e-5)
        p=expit(data[view+'_test_logits']);y=data['test_labels'];pcal=expit(data[view+'_calibration_logits'])
        cut=float(np.quantile(np.maximum(pcal,1-pcal),.2))
        np.testing.assert_allclose(cut,result['threshold'],atol=1e-12)
        keep=np.maximum(p,1-p)>=cut
        np.testing.assert_allclose(accuracy_score(y,p>=.5),result['accuracy'],atol=1e-12)
        np.testing.assert_allclose(roc_auc_score(y,p),result['auroc'],atol=1e-12)
        np.testing.assert_allclose(brier_score_loss(y,p),result['brier'],atol=1e-12)
        tn,fp,fn,tp=confusion_matrix(y,p>=.5).ravel()
        np.testing.assert_allclose(tp/(tp+fn),result['sensitivity'],atol=1e-12)
        np.testing.assert_allclose(tn/(tn+fp),result['specificity'],atol=1e-12)
        np.testing.assert_allclose(1-accuracy_score(y[keep],p[keep]>=.5),result['retained_error'],atol=1e-12)
    boot=np.load(folder/'bootstrap.npz')['native_minus_low_information']
    rng=np.random.default_rng(cfg['bootstrap_seed']);y=data['test_labels']
    for b in range(25):
        ix=rng.integers(0,len(y),len(y))
        auc=[roc_auc_score(y[ix],expit(data[v+'_test_logits'][ix])) for v in cfg['views']]
        np.testing.assert_allclose(boot[b,0],auc[0]-auc[1],atol=1e-12)
    np.testing.assert_allclose(np.quantile(boot[:,0],[.025,.975]),summary['native_minus_low_information']['auroc']['ci'],atol=1e-12)
    print('224px source/encoder/features, probe reconstruction, independent metrics and paired AUROC bootstrap verified.')


if __name__=='__main__':main()
