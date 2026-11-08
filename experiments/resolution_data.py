"""Immutable index manifest: group exact matches at either resolution."""
from pathlib import Path
import hashlib,json
import numpy as np
from sklearn.model_selection import train_test_split
ROOT=Path(__file__).resolve().parents[1]
def build():
 data={r:np.load(ROOT/'data'/('pneumoniamnist.npz' if r==28 else 'pneumoniamnist_64.npz'),allow_pickle=False) for r in [28,64]}
 records=[(s,i) for s in ['train','val','test'] for i in range(len(data[28][s+'_labels']))]
 parent=list(range(len(records)))
 def root(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 for r in [28,64]:
  seen={}
  for k,(s,i) in enumerate(records):
   h=hashlib.sha256(data[r][s+'_images'][i].tobytes()).hexdigest()
   if h in seen:parent[root(k)]=root(seen[h])
   else:seen[h]=k
 groups={}
 for k,record in enumerate(records):groups.setdefault(root(k),[]).append(record)
 selected={s:[] for s in ['train','val','test']};group_log=[]
 for members in groups.values():
  labels={int(data[28][s+'_labels'][i,0]) for s,i in members}
  assert len(labels)==1,'Conflicting labels in exact group'
  splits={s for s,i in members};assert not ('test' in splits and len(splits)>1),'Test overlap requires explicit handling'
  priority='test' if 'test' in splits else 'val' if 'val' in splits else 'train'
  rep=min(i for s,i in members if s==priority);selected[priority].append(rep)
  group_log.append({'members':members,'retained_split':priority,'retained_index':rep})
 for s in selected:selected[s].sort()
 vi=selected.pop('val');labels=data[28]['val_labels'].ravel()[vi]
 selection,calibration=train_test_split(vi,test_size=.5,stratify=labels,random_state=2026)
 selected['selection']=sorted(selection);selected['calibration']=sorted(calibration)
 result={'indices':selected,'groups':group_log,'policy':'Connected exact-pixel groups at either resolution. One representative each. Validation takes priority over training; test overlap forbidden. Validation split stratified 50/50, seed 2026. Not patient identity.','archives':{str(r):hashlib.sha256((ROOT/'data'/('pneumoniamnist.npz' if r==28 else 'pneumoniamnist_64.npz')).read_bytes()).hexdigest() for r in [28,64]}}
 (ROOT/'configs/resolution_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
 print({s:len(i) for s,i in selected.items()})
if __name__=='__main__':build()
