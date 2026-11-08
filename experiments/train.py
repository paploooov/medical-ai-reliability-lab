"""CPU pilot: validation-selected logistic and neural baselines, frozen test evaluation."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/medical-ai-mpl')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/medical-ai-cache')
import sys, json, hashlib, subprocess, platform, csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from inspect_data import main as inspect_data
import numpy as np
import scipy, sklearn
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def metrics(y,p):
    pred=p>=.5; conf=np.maximum(p,1-p); wrong=pred!=y; high=conf>=.9
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(accuracy=float(accuracy_score(y,pred)),auc=float(roc_auc_score(y,p)),brier=float(brier_score_loss(y,p)),sensitivity=float(tp/(tp+fn)),specificity=float(tn/(tn+fp)),confidence=float(conf.mean()),high_confidence_error=float((high & wrong).mean()),high_confidence_count=int(high.sum()),error_among_confident=float(wrong[high].mean()) if high.any() else None,tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))

def main():
    inspect_data()
    config=json.loads((ROOT/'configs/baseline.json').read_text())
    data=np.load(ROOT/'data/pneumoniamnist.npz',allow_pickle=False)
    def split(s): return data[s+'_images'].astype(float).reshape(-1,784)/255, data[s+'_labels'].ravel()
    x,y=split('train'); v,vy=split('val')
    candidates=[]; validation=[]
    for c in config['logistic_C']:
        m=LogisticRegression(C=c,max_iter=2000,random_state=42).fit(x,y)
        loss=log_loss(vy,m.predict_proba(v)[:,1]); candidates.append((loss,m,c))
        validation.append(dict(model='logistic',C=c,validation_log_loss=float(loss)))
    _,linear,c=min(candidates,key=lambda a:a[0])
    neural=MLPClassifier(hidden_layer_sizes=tuple(config['mlp_hidden']),random_state=42,batch_size=128,learning_rate_init=.001,alpha=.0001,max_iter=1)
    best=None; history=[]
    import copy
    for epoch in range(config['mlp_epochs']):
        neural.partial_fit(x,y,classes=np.array([0,1]))
        loss=log_loss(vy,neural.predict_proba(v)[:,1]); history.append(dict(epoch=epoch+1,validation_log_loss=float(loss)))
        if best is None or loss<best[0]: best=(loss,copy.deepcopy(neural),epoch+1)
    neural=best[1]
    out=ROOT/'reports/metrics'; out.mkdir(parents=True,exist_ok=True)
    (out/'validation.json').write_text(json.dumps(dict(logistic=validation,neural=history,selected_C=c,selected_epoch=best[2]),indent=2))
    # Save portable numeric parameters, never executable pickle files.
    np.savez_compressed(ROOT/'models/logistic.npz',coef=linear.coef_,intercept=linear.intercept_)
    np.savez_compressed(ROOT/'models/mlp.npz',w0=neural.coefs_[0],w1=neural.coefs_[1],b0=neural.intercepts_[0],b1=neural.intercepts_[1])
    tx,ty=split('test'); images=tx.reshape(-1,28,28)
    variants={'clean':images}
    for s in config['noise']: variants[f'noise_{s}']=np.clip(images+np.random.default_rng(42).normal(0,s,images.shape),0,1)
    for s in config['blur']: variants[f'blur_{s}']=gaussian_filter(images,sigma=(0,s,s),mode='reflect')
    for s in config['contrast']: variants[f'contrast_{s}']=np.clip(.5+s*(images-.5),0,1)
    rows=[]; predictions={}; rng=np.random.default_rng(123)
    for name,model in [('logistic',linear),('mlp',neural)]:
        for condition,im in variants.items():
            prob=model.predict_proba(im.reshape(-1,784))[:,1]; predictions[name+'_'+condition]=prob
            row=dict(model=name,condition=condition,**metrics(ty,prob)); rows.append(row)
        clean=(predictions[name+'_clean']>=.5)!=ty
        for row in [r for r in rows if r['model']==name]:
            p=predictions[name+'_'+row['condition']]; hc=((p>=.5)!=ty)&(np.maximum(p,1-p)>=.9)
            pc=predictions[name+'_clean']; base=clean&(np.maximum(pc,1-pc)>=.9)
            delta=hc.astype(float)-base.astype(float)
            boots=[float(delta[rng.integers(0,len(ty),len(ty))].mean()) for _ in range(2000)]
            row['paired_high_confidence_error_delta']=float(delta.mean())
            row['delta_ci_low'],row['delta_ci_high']=map(float,np.quantile(boots,[.025,.975]))
    with (out/'results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader();w.writerows(rows)
    np.savez_compressed(out/'predictions.npz',labels=ty,**predictions)
    metadata=dict(config=config,commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,data_md5=hashlib.md5((ROOT/'data/pneumoniamnist.npz').read_bytes()).hexdigest())
    (out/'run.json').write_text(json.dumps(metadata,indent=2))
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    for name in ['logistic','mlp']:
        subset=[r for r in rows if r['model']==name]
        for ax,key,title in zip(axes,['accuracy','confidence','high_confidence_error'],['Accuracy','Mean confidence','Confident errors / all images']):
            ax.plot([r['condition'] for r in subset],[r[key] for r in subset],'o-',label=name);ax.set_title(title);ax.tick_params(axis='x',rotation=70);ax.set_ylim(0,1);ax.grid(alpha=.2)
    axes[0].legend();fig.suptitle('PneumoniaMNIST • CPU pilot • 624 test images • one training seed');fig.tight_layout();fig.savefig(ROOT/'reports/figures/robustness.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,5,figsize=(12,5))
    for ax,(name,im) in zip(axes.flat,variants.items()): ax.imshow(im[0],cmap='gray',vmin=0,vmax=1);ax.set_title(name);ax.axis('off')
    fig.suptitle('Same test image under each condition; not a clinical interpretation');fig.tight_layout();fig.savefig(ROOT/'reports/figures/corruptions.png',dpi=150);plt.close(fig)
    print(json.dumps([r for r in rows if r['condition']=='clean'],indent=2))
if __name__=='__main__': main()
