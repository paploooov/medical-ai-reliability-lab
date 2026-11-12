"""Two descriptive figures from saved summaries only; no model access."""
import json,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/thoraxis-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/original_512/thoraxis_30'

def main():
    r=json.loads((OUT/'h1_results.json').read_text());m=np.array(r['contrast_table'])*100
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,(a,b)=plt.subplots(1,2,figsize=(12,4),gridspec_kw={'width_ratios':[1.15,1]})
    maximum=max(abs(m.min()),abs(m.max()));im=a.imshow(m,cmap='PiYG',vmin=-maximum,vmax=maximum,aspect='auto')
    a.set_xticks(range(5),[17,42,83,123,2026]);a.set_yticks(range(5),range(20261112,20261117));a.set_xlabel('Trainingsseed');a.set_ylabel('Splitseed')
    for i in range(5):
        for j in range(5):a.text(j,i,f'{m[i,j]:+.2f}',ha='center',va='center',color='white' if abs(m[i,j])>.6*maximum else 'black')
    a.set_title('Offiziell minus gemischt bei 80 % Abdeckung')
    fig.colorbar(im,ax=a,label='Fehlerdifferenz (Prozentpunkte)',shrink=.8)
    mean=r['primary_mean']*100
    intervals=[r['seed']['t95'],r['split']['t95'],r['group_bootstrap']['percentile95']]
    for y,(lo,hi) in enumerate(intervals):
        b.plot([lo*100,hi*100],[y,y],lw=4,color=['#277da8','#d98921','#666666'][y]);b.plot(mean,y,'ko')
    b.axvline(0,color='black',lw=1,ls='--');b.set_yticks(range(3),['Seed: 95%-t-Bereich','Split: 95%-t-Bereich','Gruppen: 95%-Bootstrap']);b.invert_yaxis();b.set_ylim(2.5,-.5)
    b.set_xlabel('Fehlerdifferenz (Prozentpunkte)');b.set_title('Drei getrennte Unsicherheitsbereiche')
    fig.text(.5,.01,'Positive Werte: mehr Fehler im offiziellen Arm. Bereiche sind kein gemeinsames Gesamtintervall.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(OUT/'h1_contrasts.svg');fig.savefig('/tmp/thoraxis-h1.png',dpi=140);plt.close(fig)
    confidence=json.loads((OUT/'h2_confidence.json').read_text())['strata'];metrics=json.loads((OUT/'h2_metrics.json').read_text())
    fig,(a,b)=plt.subplots(1,2,figsize=(12,4.2));grid=np.linspace(.5,1,501)
    for arm,color,label in [('official','#277da8','Offiziell (5 Modelle)'),('mixed','#d98921','Gemischt (25 Modelle)')]:
        rows=[x for x in confidence if x['origin']=='all' and x['stratum']=='FP' and (x['run_id'].startswith('official')==(arm=='official'))]
        curves=[]
        for row in rows:
            ecdf=row['ecdf']
            if not ecdf:continue
            xx=np.array([p['confidence'] for p in ecdf]);yy=np.r_[0,[p['cumulative_fraction'] for p in ecdf]]
            curves.append(yy[np.searchsorted(xx,grid,side='right')])
        if curves:
            curves=np.array(curves);a.plot(grid,curves.mean(0),color=color,label=label,linewidth=2)
            a.fill_between(grid,curves.min(0),curves.max(0),color=color,alpha=.15)
        rows=[x for x in metrics if x['origin']=='all' and x['cls']=='all' and (x['run_id'].startswith('official')==(arm=='official'))]
        values=[[x['fp_capture'] for x in rows if x['fp_capture'] is not None],
                [x['rejected_true_negatives']/x['tn'] for x in rows if x['tn']]]
        offset=-.17 if arm=='official' else .17
        for i,v in enumerate(values):
            if not v:continue
            v=np.array(v)*100;avg=v.mean();b.bar(i+offset,avg,.32,color=color,label=label if i==0 else None)
            b.errorbar(i+offset,avg,yerr=[[avg-v.min()],[v.max()-avg]],color='black',capsize=4)
    a.set(xlabel='Konfidenz bei falsch positiven Normalbildern',ylabel='Anteil mit höchstens dieser Konfidenz',title='Wie sicher sind die Fehlalarme?',xlim=(.5,1),ylim=(0,1));a.legend(loc='upper left',fontsize=9)
    b.set_xticks([0,1],['Fehlalarme\nabgelehnt','Richtige Normalbefunde\nabgelehnt']);b.set(ylabel='Anteil (%)',title='Ablehnung: Nutzen und Preis',ylim=(0,100));b.legend(fontsize=9)
    fig.text(.5,.01,'Gleichgewichtete Modellmittel; Bänder/Fehlerbalken zeigen Min–Max über Modelle, keine Konfidenzintervalle.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(OUT/'h2_rejection.svg');fig.savefig('/tmp/thoraxis-h2.png',dpi=140);plt.close(fig)
    # Descriptive model means/ranges, never a pooled independent case sample.
    summary=[]
    for arm in ['official','mixed']:
        for origin in ['all','train','test']:
            for cls in ['all','NORMAL','PNEUMONIA']:
                rows=[x for x in metrics if x['origin']==origin and x['cls']==cls and (x['run_id'].startswith('official')==(arm=='official'))]
                values={}
                for key in ['n','groups','specificity','sensitivity','acceptance','accepted_error','R80','R80_coverage','fp_capture','rejected_false_positives','rejected_true_negatives']:
                    v=[x[key] for x in rows if x[key] is not None]
                    values[key]=dict(mean=float(np.mean(v)),min=float(min(v)),max=float(max(v)),defined_models=len(v)) if v else None
                summary.append(dict(arm=arm,origin=origin,cls=cls,metrics=values))
    (OUT/'h2_descriptive_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (OUT/'figure_metadata.json').write_text(json.dumps(dict(matplotlib=matplotlib.__version__,figures=['h1_contrasts.svg','h2_rejection.svg'],note='No fit or inference; descriptive means and ranges'),indent=2)+'\n')
if __name__=='__main__':main()
