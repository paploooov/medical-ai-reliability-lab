"""Publication overview from frozen v1.0.0 predictions; no training or inference."""
from pathlib import Path
import argparse,csv,json,hashlib,sys,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/medical-ai-mpl')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transfer_study import variants
ROOT=Path(__file__).resolve().parents[1]
CONDITIONS=['clean','noise','blur','contrast']


def aggregate(rows,metric,arm,condition):
    selected=[r for r in rows if r['policy']=='standard' and r['arm']==arm and r['condition']==condition]
    assert sorted(int(r['seed']) for r in selected)==[17,42,83]
    values=np.array([float(r[metric]) for r in selected]) * 100
    return float(values.mean()),float(values.min()),float(values.max())


def build(out):
    for name in ["overview_figure.png", "overview_figure.pdf", "overview_summary.json", "overview_provenance.json"]:
        if (out/name).exists(): raise FileExistsError(out/name)
    out.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader((ROOT/'reports/transfer/results.csv').open()))
    pred=dict(np.load(ROOT/'reports/transfer/predictions.npz'))
    cal=dict(np.load(ROOT/'reports/calibration/predictions.npz'))
    data=np.load(ROOT/'data/pneumoniamnist.npz')
    indices=pred['test_indices']; images=variants(data['test_images'][indices].astype(np.float32)/255)
    # Same first deduplicated test case in all panels; no outcome-based selection.
    summary={arm:{c:{m:aggregate(rows,m,arm,c) for m in ['accuracy','zero_fraction','coverage','risk']} for c in CONDITIONS} for arm in ['control','augmentation']}
    cp=cal['calibration_probabilities'][1]; cc=np.maximum(cp,1-cp)
    tau=float(np.quantile(cc,.2,method='linear'))
    p=pred['control_s42_noise_probabilities']; confidence=np.maximum(p,1-p)
    zero=pred['control_s42_noise_zero']; tied=confidence==tau
    assert np.array_equal(zero,tied) and int(tied.sum())==600
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig=plt.figure(figsize=(14.8,5.1),facecolor='white');gs=fig.add_gridspec(1,3,width_ratios=[1.1,1,1.2],wspace=.32)
    left=gs[0].subgridspec(3,2,width_ratios=[1,2.2],wspace=.12,hspace=.28)
    colors=['#356391','#c26339']; labels=['Original','Noise','Contrast']
    for j,(condition,label) in enumerate(zip(['clean','noise','contrast'],labels)):
        ax=fig.add_subplot(left[j,0]);ax.imshow(images[condition][0],cmap='gray',vmin=0,vmax=1,interpolation='nearest');ax.set_axis_off();ax.set_title(label,fontsize=9,pad=3)
        ax=fig.add_subplot(left[j,1]); means=[summary[a][condition]['zero_fraction'][0] for a in summary]
        for i,a in enumerate(summary):
            mean,lo,hi=summary[a][condition]['zero_fraction'];ax.barh(i,mean,color=colors[i],height=.48)
            ax.errorbar(mean,i,xerr=[[mean-lo],[hi-mean]],color='#222222',capsize=2,fmt='none',lw=.8)
            ax.text(min(mean+2,90),i-.31,f'{mean:.1f}%',va='center',fontsize=8)
        ax.set_xlim(0,105);ax.set_ylim(1.6,-.6);ax.set_yticks([]);ax.set_xticks([0,50,100]);ax.tick_params(axis='x',labelsize=8)
        if j==2:ax.set_xlabel('Zero final hidden vector (%)',fontsize=9)
    ax=fig.add_subplot(gs[1]);bins=np.linspace(.5,1,51)
    ax.hist(confidence[~zero],bins=bins,color='#afb9c5',label='Nonzero representation')
    ax.hist(confidence[zero],bins=bins,color=colors[0],label='Zero representation')
    ax.axvline(tau,color='#222222',linestyle='--',lw=1)
    ax.annotate('600 / 621 cases\n$h=0 \\Rightarrow z=b$\nPrediction: normal',xy=(tau,600),xytext=(.51,450),arrowprops={'arrowstyle':'-','lw':.8},fontsize=9)
    ax.set(xlim=(.5,1),ylim=(0,720),xlabel='Confidence under noise',ylabel='Test cases')
    ax.text(.51,690,f'Control, seed 42  |  $\\tau$ = {tau:.6f}',va='top',fontsize=9)
    ax.text(.80,340,'Calibration\n90.08% accepted\nTarget: 80%\n\nNoise test\n96.94% accepted',fontsize=8)
    ax=fig.add_subplot(gs[2])
    for i,(arm,label) in enumerate([('control','Control'),('augmentation','Noise augmentation')]):
        vals=np.array([summary[arm][c]['risk'] for c in CONDITIONS]);x=np.arange(4)+(i-.5)*.18
        ax.errorbar(x,vals[:,0],yerr=[vals[:,0]-vals[:,1],vals[:,2]-vals[:,0]],fmt='o',color=colors[i],capsize=4,lw=1.4,label=label,markersize=5)
    ax.set(xticks=range(4),xticklabels=['Original','Noise','Blur','Contrast'],ylabel='Error among accepted cases (%)',ylim=(0,70))
    ax.grid(axis='y',alpha=.15);ax.legend(frameon=False,loc='upper right',fontsize=9)
    ax.text(.02,.94,'Original calibration thresholds ($\\geq\\tau$)',transform=ax.transAxes,fontsize=8,va='top')
    for x,title in [( .055,'A  Input & representation'),(.386,'B  Bias-only confidence spike'),(.701,'C  Transfer beyond noise')]:fig.text(x,.96,title,weight='bold',fontsize=11)
    fig.subplots_adjust(top=.88,bottom=.18,left=.045,right=.985)
    fig.text(.045,.065,'Blue: control   ·   Orange: noise augmentation   |   A, C: mean and min–max of 3 seeds; not confidence intervals.',fontsize=9)
    fig.text(.045,.025,f'PneumoniaMNIST test index {int(indices[0])}: same case in all three images (CC BY 4.0). 28 × 28 pixels; fixed models and test split.',fontsize=8,color='#555555')
    fig.savefig(out/'overview_figure.png',dpi=300)
    fig.savefig(out/'overview_figure.pdf',metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)
    paths=['reports/transfer/results.csv','reports/transfer/predictions.npz','reports/calibration/predictions.npz','data/pneumoniamnist.npz','scripts/make_overview.py','experiments/transfer_study.py']
    (out/'overview_summary.json').write_text(json.dumps(dict(metrics_percent_mean_min_max=summary,example_test_index=int(indices[0]),panel_b=dict(seed=42,tau=tau,calibration_accepted=int((cc>=tau).sum()),calibration_cases=len(cc),noise_zero=int(zero.sum()),noise_accepted=int((confidence>=tau).sum()))),indent=2)+'\n')
    (out/'overview_provenance.json').write_text(json.dumps({p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'reports/figures');build(parser.parse_args().output)
