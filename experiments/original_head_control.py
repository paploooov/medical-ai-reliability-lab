"""Matched head-only optimizer control, with no additional encoder adaptation."""
import copy,json,time
import numpy as np
import torch
from scipy.special import expit
from original_512 import ROOT,CACHE,vector
from prepare_original_xray import digest
from high_resolution_probe import evaluate
from high_resolution_robustness import interval
from temperature_scaling import nll

def main():
    torch.set_num_threads(1);torch.manual_seed(20261121);parent=ROOT/'reports/original_512';out=parent/'head_control';out.mkdir(exist_ok=False)
    while not (parent/'probe_heads.npz').exists():time.sleep(10)
    manifest=json.loads((parent/'manifest.json').read_text());records=manifest['records'];y=np.array([r['label'] for r in records]);ix={s:np.array([i for i,r in enumerate(records) if r['split']==s]) for s in ['train','selection','calibration','test']};f=torch.tensor(np.load(CACHE/'mobile_512_features.npz')['features'])
    original=np.load(parent/'probe_heads.npz');head=torch.nn.Linear(576,1)
    with torch.no_grad():
        head.weight.copy_(torch.tensor(original['mobile_512_probe_weight'][None],dtype=torch.float32));head.bias.copy_(torch.tensor([float(original['mobile_512_probe_bias'])],dtype=torch.float32))
    def predict():
        with torch.inference_mode():return head(f).ravel().numpy().copy()
    z=predict();best_loss=nll(y[ix['selection']],z[ix['selection']]);best=copy.deepcopy(head.state_dict());selected=0;history=[dict(epoch=0,selection_nll=best_loss)];rng=np.random.default_rng(20261121);opt=torch.optim.AdamW(head.parameters(),lr=1e-4,weight_decay=1e-4)
    for epoch in range(1,4):
        order=rng.permutation(ix['train']);total=0
        for start in range(0,len(order),16):
            ids=order[start:start+16];opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(head(f[ids]).ravel(),torch.tensor(y[ids],dtype=torch.float32));loss.backward();opt.step();total+=float(loss.detach())*len(ids)
        z=predict();val=nll(y[ix['selection']],z[ix['selection']]);history.append(dict(epoch=epoch,training_nll=total/len(order),selection_nll=val))
        if val<best_loss:best_loss=val;best=copy.deepcopy(head.state_dict());selected=epoch
    head.load_state_dict(best);z=predict();pc=expit(z[ix['calibration']]);cut=float(np.quantile(np.maximum(pc,1-pc),.2));np.savez_compressed(out/'head_and_predictions.npz',logits=z,weight=head.weight.detach().numpy(),bias=head.bias.detach().numpy());print('Head control selected',selected,history,flush=True)
    while not (parent/'summary.json').exists():time.sleep(10)
    base=np.load(parent/'predictions.npz');fine=base['mobile_512_finetuned'];pc=expit(fine[ix['calibration']]);fcut=float(np.quantile(np.maximum(pc,1-pc),.2));test=ix['test'];yt=y[test];groups=np.array([records[i]['group'] for i in test]);members=[np.flatnonzero(groups==g) for g in np.unique(groups)];rng=np.random.default_rng(20261122);hb=[];fb=[]
    for _ in range(2000):
        ids=np.concatenate([members[i] for i in rng.integers(0,len(members),len(members))]);hb.append(vector(yt[ids],z[test][ids],cut));fb.append(vector(yt[ids],fine[test][ids],fcut))
    hb=np.array(hb);fb=np.array(fb);point=vector(yt,fine[test],fcut)-vector(yt,z[test],cut);metrics=['auroc','nll','accuracy','risk_at_80','coverage','retained_error'];summary=dict(config=json.loads((ROOT/'configs/original_512_head_control.json').read_text()),selected_epoch=selected,history=history,metrics=evaluate(yt,z[test],cut),fine_minus_head={m:interval(point[i],fb[:,i]-hb[:,i]) for i,m in enumerate(metrics)},script_sha256=digest(__file__),initial_head_sha256=digest(parent/'probe_heads.npz'))
    np.savez_compressed(out/'bootstrap.npz',head=hb,fine=fb);(out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');print('COMPLETE head control',summary['fine_minus_head']['auroc'],flush=True)

if __name__=='__main__':main()
