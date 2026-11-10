"""Compact manuscript figure from saved frozen-probe predictions."""
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
    folder=ROOT/'reports/high_resolution/robustness'
    saved=np.load(folder/'predictions.npz');y=saved['labels']
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharex=True,sharey=True)
    for ax,condition,title in zip(axes,['clean','noise_input'],['Clean images','Noise placement control']):
        for view,color,name in [('native_224','#345995','224px information'),('downsample28_upsample224','#bd5735','28px information')]:
            ax.plot(COVERAGES,risk_coverage(y,expit(saved[view+'__'+condition])),color=color,label=name+(' / input noise' if condition=='noise_input' else ''))
        ax.set(title=title,xlabel='Coverage',ylim=(0,.42));ax.axvline(.8,color='gray',linestyle=':');ax.grid(alpha=.2)
    axes[1].plot(COVERAGES,risk_coverage(y,expit(saved['downsample28_upsample224__noise_source'])),color='#bd5735',linestyle='--',label='28px / source noise (filtered)')
    axes[0].set_ylabel('Error among retained cases')
    for ax in axes:ax.legend(fontsize=7,loc='upper left')
    fig.suptitle('Frozen probes: clean advantage does not establish greater noise resistance')
    fig.tight_layout();fig.savefig(folder/'summary_figure.png',dpi=180);fig.savefig(folder/'summary_figure.pdf');plt.close(fig)


if __name__=='__main__':main()
