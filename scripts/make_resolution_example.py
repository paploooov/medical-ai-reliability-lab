"""Show a prespecified training case at 224px and reduced to 28px; no test selection."""
from pathlib import Path
import os,sys,json
os.environ.setdefault('MPLCONFIGDIR','/tmp/medical-ai-mpl')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from high_resolution_probe import low_information


def main():
    out=ROOT/'reports/high_resolution';out.mkdir(exist_ok=True)
    idx=json.loads((ROOT/'configs/resolution_manifest.json').read_text())['indices']['train'][0]
    data=np.load(ROOT/'data/pneumoniamnist_224.npz');original=data['train_images'][idx]
    low=low_information(original)
    fig,axes=plt.subplots(1,2,figsize=(8,4.5))
    for ax,image,title in zip(axes,[original,low],['224 × 224 information','28 × 28 information, displayed at 224']):
        ax.imshow(image,cmap='gray',vmin=0,vmax=255,interpolation='nearest');ax.set_title(title,fontsize=11);ax.axis('off')
    fig.suptitle(f'Fixed example: first retained training case (original index {idx})')
    fig.text(.5,.02,'PneumoniaMNIST / MedMNIST, CC BY 4.0; centre-cropped and resized source radiograph.\nRight: additional bicubic downsampling and upsampling; illustration, not diagnostic assessment.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.09,1,.94]);fig.savefig(out/'resolution_example.png',dpi=180);fig.savefig(out/'resolution_example.pdf');plt.close(fig)

if __name__=='__main__':main()
