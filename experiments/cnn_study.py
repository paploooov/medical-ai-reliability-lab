"""Three-seed CNN follow-up; validation-only checkpoint, calibration and rejection."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/medical-ai-mpl')
from pathlib import Path
import json, csv, copy, hashlib, subprocess
import numpy as np
import torch
from torch import nn
from scipy.special import expit
from scipy.optimize import minimize_scalar
from scipy.ndimage import gaussian_filter
from sklearn.metrics import log_loss
from train import metrics
from error_detection import detection, risk_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/cnn'
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Flatten(),nn.Linear(32*7*7,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self,x):return self.layers(x).squeeze(1)

def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    config=json.loads((ROOT/'configs/cnn_study.json').read_text())
    data=np.load(ROOT/'data/pneumoniamnist.npz',allow_pickle=False)
    def split(s):return torch.tensor(data[s+'_images'][:,None]/255,dtype=torch.float32),torch.tensor(data[s+'_labels'].ravel(),dtype=torch.float32)
    x,y=split('train');v,vy=split('val');test,ty=split('test'); labels=ty.numpy()
    rows=[];history=[];all_predictions={};choices=[]
    images=test.numpy()[:,0]
    variants={'clean':images}
    for s in [.05,.1,.2]:variants[f'noise_{s}']=np.clip(images+np.random.default_rng(42).normal(0,s,images.shape),0,1)
    for s in [.5,1,2]:variants[f'blur_{s}']=gaussian_filter(images,sigma=(0,s,s),mode='reflect')
    for s in [.75,.5,.25]:variants[f'contrast_{s}']=np.clip(.5+s*(images-.5),0,1)
    for seed in config['seeds']:
        torch.manual_seed(seed);model=CNN();opt=torch.optim.Adam(model.parameters(),lr=config['learning_rate']);lossfn=nn.BCEWithLogitsLoss()
        best=None
        for epoch in range(1,config['epochs']+1):
            model.train();order=torch.randperm(len(x));total=0
            for batch in order.split(config['batch_size']):
                opt.zero_grad();loss=lossfn(model(x[batch]),y[batch]);loss.backward();opt.step();total+=loss.item()*len(batch)
            model.eval()
            with torch.no_grad():val_loss=lossfn(model(v),vy).item()
            history.append(dict(seed=seed,epoch=epoch,train_loss=total/len(x),validation_loss=val_loss))
            if best is None or val_loss<best[0]:best=(val_loss,copy.deepcopy(model.state_dict()),epoch)
            print(f'seed={seed} epoch={epoch:02d} train={total/len(x):.4f} validation={val_loss:.4f}',flush=True)
        model.load_state_dict(best[1]);model.eval()
        with torch.no_grad():vz=model(v).numpy()
        fit=minimize_scalar(lambda t:log_loss(vy.numpy(),expit(vz/np.exp(t))),bounds=(-3,3),method='bounded')
        temperature=float(np.exp(fit.x))
        choices.append(dict(seed=seed,epoch=best[2],validation_loss=best[0],temperature=temperature,parameters=sum(p.numel() for p in model.parameters())))
        torch.save(model.state_dict(),ROOT/f'models/cnn/seed_{seed}.pt')
        for mode,T in [('raw',1.),('temperature',temperature)]:
            vp=expit(vz/T);threshold=float(np.quantile(np.maximum(vp,1-vp),.2))
            for condition,im in variants.items():
                with torch.no_grad():p=expit(model(torch.tensor(im[:,None],dtype=torch.float32)).numpy()/T)
                all_predictions[f'{seed}_{mode}_{condition}']=p
                rows.append(dict(seed=seed,mode=mode,condition=condition,threshold=threshold,**metrics(labels,p),**detection(labels,p,threshold)))
        (OUT/f'progress_seed_{seed}.json').write_text(json.dumps(choices[-1],indent=2))
    for name,records in [('results',rows),('learning_curves',history)]:
        with (OUT/f'{name}.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
    np.savez_compressed(OUT/'predictions.npz',labels=labels,**all_predictions)
    (OUT/'run.json').write_text(json.dumps(dict(config=config,choices=choices,torch=torch.__version__,numpy=np.__version__,commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),data_sha256=hashlib.sha256((ROOT/'data/pneumoniamnist.npz').read_bytes()).hexdigest()),indent=2))
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
    for ax,seed in zip(axes,config['seeds']):
        h=[r for r in history if r['seed']==seed]
        ax.plot([r['epoch'] for r in h],[r['train_loss'] for r in h],label='Training');ax.plot([r['epoch'] for r in h],[r['validation_loss'] for r in h],label='Validation');ax.set(title=f'Seed {seed}',xlabel='Epoch',ylabel='Binary cross-entropy');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'learning_curves.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,key,title in zip(axes,['accuracy','high_confidence_error','error_detection_auroc'],['Accuracy','Confident errors / all images','Error-detection AUROC']):
        for mode,color in [('raw','#345995'),('temperature','#d1603d')]:
            values=np.array([[next(r[key] for r in rows if r['seed']==seed and r['mode']==mode and r['condition']==c) for c in variants] for seed in config['seeds']])
            means=values.mean(0);ax.plot(list(variants),means,'o-',color=color,label=mode);ax.fill_between(list(variants),values.min(0),values.max(0),color=color,alpha=.15)
        ax.set(title=title,ylim=(0,1));ax.tick_params(axis='x',rotation=75);ax.grid(alpha=.2)
    axes[0].legend();fig.suptitle('CNN: mean and min–max across three seeds (not confidence intervals)');fig.tight_layout();fig.savefig(OUT/'robustness.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for mode in ['raw','temperature']:
        for seed in config['seeds']:
            p=all_predictions[f'{seed}_{mode}_clean'];coverage,risk=risk_curve(labels,p);axes[1].plot(coverage,risk,alpha=.5,label=f'{mode} {seed}')
        # Calibration points from all seeds; descriptive pooling, not independent observations.
        p=np.concatenate([all_predictions[f'{seed}_{mode}_clean'] for seed in config['seeds']]); yy=np.tile(labels,len(config['seeds']))
        xx=[];zz=[]
        for lo in np.linspace(0,.9,10):
            keep=(p>=lo)&(p<lo+.1 if lo<.9 else p<=1)
            if keep.any():xx.append(p[keep].mean());zz.append(yy[keep].mean())
        axes[0].plot(xx,zz,'o-',label=mode)
    axes[0].plot([0,1],[0,1],'k--');axes[0].set(title='Clean calibration: pooled seeds',xlabel='Predicted pneumonia probability',ylabel='Observed positive fraction');axes[0].legend()
    axes[1].set(title='Clean selective prediction',xlabel='Coverage',ylabel='Retained error rate');axes[1].legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'calibration_selection.png',dpi=180)
if __name__=='__main__':main()
