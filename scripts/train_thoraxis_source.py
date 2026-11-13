"""Five grouped OOF source classifiers on NORMAL only, with inner selection."""
import copy,json,sys,time
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from scipy.special import expit
from prepare_thoraxis_cache import verify,encoder,records
from thoraxis_stage3_common import ROOT,OUT,guard,save,digest
FOLDS=ROOT/'configs/thoraxis_source_folds.json'

def prepare():
    gate=guard();rr=[r for r in records() if r['label']==0];y=np.array([r['original_split']=='test' for r in rr],dtype=int);groups=np.array([r['group'] for r in rr]);folds=[]
    outer=StratifiedGroupKFold(5,shuffle=True,random_state=20261113)
    for fold,(rest,test) in enumerate(outer.split(np.zeros(len(rr)),y,groups)):
        inner=StratifiedGroupKFold(5,shuffle=True,random_state=20261114)
        tr,sel=next(inner.split(np.zeros(len(rest)),y[rest],groups[rest]));train=rest[tr];selection=rest[sel]
        for ids in (train,selection,test):assert set(y[ids])=={0,1},'Missing source class; stop'
        sets=[set(groups[ix]) for ix in (train,selection,test)];assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
        folds.append(dict(fold=fold,train=train.tolist(),selection=selection.tolist(),test=test.tolist(),counts={s:np.bincount(y[ix],minlength=2).tolist() for s,ix in [('train',train),('selection',selection),('test',test)]}))
    assert sorted(i for f in folds for i in f['test'])==list(range(len(rr)))
    assert not FOLDS.exists()
    save(FOLDS,dict(gate=gate,source_manifest_sha256=digest(ROOT/'reports/original_512/manifest.json'),records=rr,folds=folds))
    print('Fold manifest prepared',len(rr),flush=True)

def train():
    gate=guard();import subprocess
    assert subprocess.check_output(['git','show',gate['remote_main']+':configs/thoraxis_source_folds.json'],cwd=ROOT)==FOLDS.read_bytes()
    target=ROOT/'data/thoraxis_source_shift/source_cnn';target.mkdir(parents=True,exist_ok=False)
    folds=json.loads(FOLDS.read_text());rr=folds['records'];allrows=records();index={r['name']:i for i,r in enumerate(allrows)};y=np.array([r['original_split']=='test' for r in rr],dtype=int)
    groups=np.array([r['group'] for r in rr]);oof=np.full(len(rr),np.nan,dtype=np.float32);outerfold=np.full(len(rr),-1);runs=[]
    torch.set_num_threads(1)
    for f in folds['folds']:
        start=time.monotonic();torch.manual_seed(17);np.random.seed(17);_,arrays=verify()
        cache=arrays['prefix.npy'][[index[r['name']] for r in rr]].copy();del arrays
        model=encoder();tail=torch.nn.Sequential(model.features[10:],torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten());head=torch.nn.Linear(576,1)
        for p in tail.parameters():p.requires_grad_(True)
        for module in tail.modules():
            if isinstance(module,torch.nn.BatchNorm2d):
                for p in module.parameters():p.requires_grad_(False)
        net=torch.nn.Sequential(tail,head).eval();tr=np.array(f['train']);sel=np.array(f['selection']);test=np.array(f['test'])
        def predict(ids):
            with torch.inference_mode():return np.concatenate([net(torch.tensor(cache[ids[i:i+32]].copy())).ravel().numpy() for i in range(0,len(ids),32)])
        def nll(z):return float(np.mean(np.logaddexp(0,z)-y[sel]*z))
        best_loss=nll(predict(sel));best=copy.deepcopy(net.state_dict());best_epoch=0;history=[dict(epoch=0,selection_nll=best_loss)]
        weights=len(tr)/(2*np.bincount(y[tr],minlength=2));rng=np.random.default_rng(17)
        optimizer=torch.optim.AdamW([{'params':[p for p in tail.parameters() if p.requires_grad],'lr':1e-5},{'params':head.parameters(),'lr':1e-4}],weight_decay=1e-4)
        for epoch in range(1,4):
            t=time.monotonic();order=rng.permutation(tr);loss_sum=0.
            for first in range(0,len(order),16):
                ids=order[first:first+16];optimizer.zero_grad();z=net(torch.tensor(cache[ids].copy())).ravel()
                losses=torch.nn.functional.binary_cross_entropy_with_logits(z,torch.tensor(y[ids],dtype=torch.float32),reduction='none')
                loss=(losses*torch.tensor(weights[y[ids]],dtype=torch.float32)).mean();loss.backward();optimizer.step();loss_sum+=float(loss.detach())*len(ids)
            val=nll(predict(sel));history.append(dict(epoch=epoch,weighted_train_loss=loss_sum/len(tr),selection_nll=val,seconds=time.monotonic()-t))
            if val<best_loss:best_loss=val;best=copy.deepcopy(net.state_dict());best_epoch=epoch
            print('Source fold',f['fold'],'epoch',epoch,flush=True)
        net.load_state_dict(best);oof[test]=predict(test);outerfold[test]=f['fold'];model.classifier=head
        path=target/f"fold_{f['fold']}.pt";torch.save(dict(state_dict=model.state_dict(),input_size=512,source_label='original official test = 1',fold=f['fold'],selected_epoch=best_epoch),path)
        runs.append(dict(fold=f['fold'],history=history,selected_epoch=best_epoch,selected_nll=best_loss,class_weights=weights.tolist(),seconds=time.monotonic()-start,checkpoint_path=str(path.relative_to(ROOT)),checkpoint_sha256=digest(path)))
        save(OUT/'source_cnn_runs.json',dict(gate=gate,fold_manifest_sha256=digest(FOLDS),runs=runs,complete=len(runs)==5))
    assert np.isfinite(oof).all() and np.all(outerfold>=0)
    with (OUT/'source_oof.npz').open('xb') as f:
        np.savez_compressed(f,image_id=np.array([r['name'] for r in rr]),group=groups,label=y,fold=outerfold,logit=oof,score=expit(oof))
    save(OUT/'source_oof_manifest.json',dict(file='source_oof.npz',sha256=digest(OUT/'source_oof.npz'),n=len(rr),counts=np.bincount(y).tolist(),fold_manifest_sha256=digest(FOLDS)))
    print('All source OOF predictions saved',flush=True)
if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='train':train()
    else:raise ValueError('Choose prepare or train')
