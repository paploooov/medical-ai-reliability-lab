"""Compact primary/clean comparison from the saved intervention results."""
from pathlib import Path
import os,sys
os.environ.setdefault('MPLCONFIGDIR','/tmp/medical-ai-mpl')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
import numpy as np
from scipy.special import expit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from external_eval import risk_coverage,COVERAGES


def main():
    folder=ROOT/'reports/high_resolution/augmentation'
    data=np.load(folder/'predictions.npz');y=data['labels']
    fig,axes=plt.subplots(1,2,figsize=(9,3.8),sharex=True,sharey=True)
    curves=[]
    for ax,condition,title in zip(axes,['clean','noise_42'],['Clean: tradeoff check','Input noise: prespecified primary']):
        for arm,color,label in [('control','#345995','Clean-trained head'),('augmentation','#bd5735','Noise-augmented head')]:
            risk=risk_coverage(y,expit(data[arm+'__'+condition]));curves.append(risk)
            ax.plot(COVERAGES,risk,color=color,label=label)
        ax.set(title=title,xlabel='Coverage');ax.axvline(.8,color='gray',linestyle=':');ax.grid(alpha=.2)
    upper=max(.2,float(np.max(curves))*1.1)
    axes[0].set(ylabel='Error among retained cases',ylim=(0,min(1,upper)))
    axes[0].legend(fontsize=8)
    fig.suptitle('Same frozen 224px encoder; only the classification head is retrained')
    fig.tight_layout();fig.savefig(folder/'summary_figure.png',dpi=180);fig.savefig(folder/'summary_figure.pdf');plt.close(fig)


if __name__=='__main__':main()
