"""Summarize paired source-resolution and training results without winner filtering."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/medical-ai-mpl')
from pathlib import Path
import json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
from external_eval import risk_at

def main():
    out=ROOT/'reports/original_512';manifest=json.loads((out/'manifest.json').read_text());records=manifest['records'];i=next(i for i,r in enumerate(records) if r['split']=='train')
    low=np.load(ROOT/'data/original_xray/images_224.npy',mmap_mode='r')[i];high=np.load(ROOT/'data/original_xray/images_512.npy',mmap_mode='r')[i]
    fig,axes=plt.subplots(1,2,figsize=(9,4.5))
    for ax,image,title in zip(axes,[np.asarray(Image.fromarray(low).resize((512,512),Image.Resampling.BICUBIC)),high],['224px representation, enlarged for display','512px representation from original']):
        ax.imshow(image,cmap='gray',vmin=0,vmax=255);ax.set_title(title,fontsize=10);ax.axis('off')
    r=records[i];fig.suptitle(f"Same training image; original {r['width']} × {r['height']} px",fontsize=11)
    fig.text(.5,.015,'Kermany, Zhang & Goldbaum (2018), Mendeley rscbjbr9sj.2 · CC BY 4.0\nModified: grayscale, aspect-preserving resize and black padding.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.07,1,.95]);fig.savefig(out/'resolution_example.png',dpi=160);plt.close(fig)
    if not (out/'summary.json').exists():return
    s=json.loads((out/'summary.json').read_text());p=np.load(out/'predictions.npz');test=p['test_indices'];y=p['labels'][test]
    names={'mobile_224_probe':'Mobile 224 frozen','mobile_512_probe':'Mobile 512 frozen','mobile_512_finetuned':'Mobile 512 adapted','resnet18_224_probe':'ResNet18 224 frozen'}
    logits={a:p[a][test] for a in names};auc={a:s['results'][a]['auroc'] for a in names}
    if (out/'large_encoder/summary.json').exists():
        large=json.loads((out/'large_encoder/summary.json').read_text());lp=np.load(out/'large_encoder/head_and_predictions.npz');lb=np.load(out/'large_encoder/bootstrap.npz')
        names['resnet50_224_probe']='ResNet50 224 frozen';logits['resnet50_224_probe']=lp['logits'][test];auc['resnet50_224_probe']=dict(point=large['metrics']['auroc'],ci=np.quantile(lb['resnet50'][:,0],[.025,.975]))
    from scipy.special import expit
    fig,axes=plt.subplots(1,3,figsize=(13,3.8));coverage=np.linspace(.2,1,81)
    maxrisk=0.
    for a,name in names.items():
        prob=expit(logits[a]);curve=[risk_at(y,prob,c) for c in coverage];maxrisk=max(maxrisk,max(curve));axes[0].plot(coverage,curve,label=name)
    axes[0].set(xlabel='Coverage',ylabel='Error among accepted',ylim=(0,min(1,max(.1,1.15*maxrisk))));axes[0].legend(fontsize=7)
    h=s['selection']['mobile_512_finetuned']['history'];axes[1].plot([v['epoch'] for v in h],[v['selection_nll'] for v in h],marker='o');axes[1].set(xlabel='Training epoch',ylabel='Selection NLL',xticks=[0,1,2,3]);axes[1].axvline(s['selection']['mobile_512_finetuned']['selected_epoch'],color='gray',ls='--')
    for j,(a,name) in enumerate(names.items()):
        v=auc[a];axes[2].errorbar(v['point'],j,xerr=[[v['point']-v['ci'][0]],[v['ci'][1]-v['point']]],fmt='o',capsize=3)
    axes[2].set(yticks=range(len(names)),yticklabels=list(names.values()),xlabel='Test AUROC, grouped bootstrap 95%');axes[2].tick_params(axis='y',labelsize=8)
    fig.tight_layout();fig.savefig(out/'model_comparison.png',dpi=180);fig.savefig(out/'model_comparison.pdf');plt.close(fig)

if __name__=='__main__':main()
