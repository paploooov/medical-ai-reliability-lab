"""Verify official 64px archive, labels and pixel overlaps; visualize same train cases."""
from pathlib import Path
import os,hashlib,json,urllib.request
os.environ.setdefault('MPLCONFIGDIR','/tmp/medical-ai-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def main():
 path=ROOT/'data/pneumoniamnist_64.npz';expected='8f4eceb4ccffa70c672198ea285246c6'
 if not path.exists():
  tmp=path.with_suffix('.download');urllib.request.urlretrieve('https://zenodo.org/records/10519652/files/pneumoniamnist_64.npz?download=1',tmp)
  assert hashlib.md5(tmp.read_bytes()).hexdigest()==expected
  tmp.replace(path)
 assert hashlib.md5(path.read_bytes()).hexdigest()==expected
 small=np.load(ROOT/'data/pneumoniamnist.npz',allow_pickle=False);large=np.load(path,allow_pickle=False)
 hashes={}
 for split,count in [('train',4708),('val',524),('test',624)]:
  assert large[split+'_images'].shape==(count,64,64)
  np.testing.assert_array_equal(small[split+'_labels'],large[split+'_labels'])
  hashes[split]=[hashlib.sha256(im.tobytes()).hexdigest() for im in large[split+'_images']]
 result={'resolution':64,'md5':expected,'labels_match_28px':True,'overlaps':{a+'_'+b:len(set(hashes[a])&set(hashes[b])) for a,b in [('train','val'),('train','test'),('val','test')]},'duplicate_excess':{s:len(h)-len(set(h)) for s,h in hashes.items()}}
 (ROOT/'reports/audit/resolution_64.json').write_text(json.dumps(result,indent=2)+'\n')
 fig,axes=plt.subplots(2,4,figsize=(10,6))
 for col,index in enumerate([0,1,2,3]):
  for row,(res,d) in enumerate([(28,small),(64,large)]):
   axes[row,col].imshow(d['train_images'][index],cmap='gray',vmin=0,vmax=255,interpolation='nearest');axes[row,col].axis('off');axes[row,col].set_title(f'Train #{index} · {res}×{res}')
 fig.suptitle('Same source cases, independently resized by MedMNIST\nPublic pediatric X-rays; illustration, not diagnostic interpretation');fig.tight_layout();fig.savefig(ROOT/'reports/audit/resolution_comparison.png',dpi=160)
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
