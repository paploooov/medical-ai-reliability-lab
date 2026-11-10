"""Plot all severity conditions for the original and targeted intervention heads."""
from pathlib import Path
import os,json
os.environ.setdefault('MPLCONFIGDIR','/tmp/medical-ai-mpl')
ROOT=Path(__file__).resolve().parents[1]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    folder=ROOT/'reports/high_resolution/mixed_augmentation';s=json.loads((folder/'summary.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,family,levels in zip(axes,['noise','blur'],[[0,.025,.05,.1,.15,.2],[0,2,4,8,16]]):
        names=['clean' if v==0 else f'{family}_{v}' for v in levels]
        for arm,color,style in [('control','#7b8794',':'),('noise','#bd5735','-'),('reduced_noise','#8c6bb1','--'),('mixed','#2a8068','-')]:
            values=[s['results'][arm+'|'+name]['risk_at_80']['point'] for name in names]
            ax.plot(levels,values,style,marker='o',color=color,label=arm)
        ax.set(title=family,xlabel='Noise SD' if family=='noise' else 'Blur sigma (pixels)',ylim=(0,1));ax.grid(alpha=.2)
    axes[0].set_ylabel('Error at 80% coverage');axes[0].legend(fontsize=8)
    fig.suptitle('Targeted mixture versus noise-weight control; fixed encoder, known internal benchmark')
    fig.tight_layout();fig.savefig(folder/'severity.png',dpi=180);fig.savefig(folder/'severity.pdf');plt.close(fig)


if __name__=='__main__':main()
