"""Render the frozen preprocessing comparison from saved results."""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/medical-ai-mpl")
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def main():
    out=ROOT/'reports/original_512/preprocessing'
    s=json.loads((out/'summary.json').read_text())['results']
    keys=list(s);names=['Lanczos reference','Bilinear','Center crop','Stretch','Gamma 0.8','Gamma 1.25'];y=np.arange(len(keys))
    fig,axes=plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
    p=np.array([s[k]['flip_fraction']['point'] for k in keys])*100
    ci=np.array([s[k]['flip_fraction']['ci'] for k in keys])*100
    axes[0].errorbar(p,y,xerr=np.stack([p-ci[:,0],ci[:,1]-p]),fmt='o',capsize=3,color='#24618d')
    axes[0].set(yticks=y,yticklabels=names,xlabel='Changed decisions (%)',title='Paired changes; 95% group bootstrap')
    for metric,label,color in [('sensitivity','Sensitivity','#24618d'),('specificity','Specificity','#b76025')]:
        axes[1].plot([s[k][metric]['point']*100 for k in keys],y,'o',label=label,color=color)
    axes[1].set(yticks=y,yticklabels=names,xlim=(35,102),xlabel='Percent',title='Same model and decision threshold')
    axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.15),ncol=2)
    for ax in axes:ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    fig.suptitle('512px preprocessing stress: 618 known test images')
    fig.savefig(out/'comparison.png',dpi=170);fig.savefig(out/'comparison.pdf');plt.close(fig)
if __name__=='__main__':main()
