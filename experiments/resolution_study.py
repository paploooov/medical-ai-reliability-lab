"""Fixed protocol resolution study. Run after manifest generation."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/medical-ai-mpl')
import json,csv,copy,time,hashlib,subprocess
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.special import expit
from scipy.ndimage import gaussian_filter
from train import metrics
from error_detection import detection
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/resolution'
class ResolutionCNN(nn.Module):
 def __init__(self):
  super().__init__();self.layers=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.AdaptiveAvgPool2d((4,4)),nn.Flatten(),nn.Linear(512,32),nn.ReLU(),nn.Linear(32,1))
 def forward(self,x):return self.layers(x).squeeze(1)
def logits(model,x):
 with torch.no_grad():return torch.cat([model(b) for b in x.split(128)]).numpy()
def tensor(x):return torch.tensor(x[:,None]/255,dtype=torch.float32)
def main(config_path=ROOT/"configs/resolution_study.json", output_dir=OUT, model_dir=ROOT/"models/resolution"):
 OUT=Path(output_dir);OUT.mkdir(parents=True,exist_ok=True)
 model_dir=Path(model_dir);model_dir.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
 config=json.loads(Path(config_path).read_text());manifest=json.loads((ROOT/'configs/resolution_manifest.json').read_text());idx=manifest['indices'];rows=[];history=[];choices=[];pred={}
 started=time.perf_counter()
 for res in config['resolutions']:
  path=ROOT/'data'/('pneumoniamnist.npz' if res==28 else 'pneumoniamnist_64.npz');assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['archives'][str(res)]
  data=np.load(path)
  x=tensor(data['train_images'][idx['train']]);y=torch.tensor(data['train_labels'][idx['train']].ravel(),dtype=torch.float32)
  vx=tensor(data['val_images'][idx['selection']]);vy=data['val_labels'][idx['selection']].ravel();cx=tensor(data['val_images'][idx['calibration']])
  images=data['test_images'][idx['test']].astype(np.float32)/255;labels=data['test_labels'][idx['test']].ravel()
  variants={'clean':images,'relative_blur':gaussian_filter(images,(0,res/28,res/28),mode='reflect'),'fixed_blur':gaussian_filter(images,(0,1,1),mode='reflect'),'noise':np.clip(images+np.random.default_rng(42).normal(0,.1,images.shape),0,1),'contrast':np.clip(.5+.5*(images-.5),0,1)}
  for seed in config['seeds']:
   torch.manual_seed(seed);model=ResolutionCNN();optim=torch.optim.Adam(model.parameters(),lr=config['learning_rate']);lossfn=nn.BCEWithLogitsLoss();best=None;start=time.perf_counter()
   for epoch in range(1,config['epochs']+1):
    model.train();total=0
    for batch in torch.randperm(len(x)).split(config['batch_size']):
     optim.zero_grad();loss=lossfn(model(x[batch]),y[batch]);loss.backward();optim.step();total+=loss.item()*len(batch)
    model.eval();vl=logits(model,vx);v_loss=float(np.mean(np.logaddexp(0,vl)-vy*vl))
    history.append(dict(resolution=res,seed=seed,epoch=epoch,training_online_bce=total/len(x),selection_bce=v_loss))
    if best is None or v_loss<best[0]:best=(v_loss,copy.deepcopy(model.state_dict()),epoch)
    print(f'res={res} seed={seed} epoch={epoch} train={total/len(x):.4f} selection={v_loss:.4f}',flush=True)
   seconds=time.perf_counter()-start;model.load_state_dict(best[1]);model.eval();cp=expit(logits(model,cx));cut=float(np.quantile(np.maximum(cp,1-cp),.2))
   path=model_dir/f'r{res}_s{seed}.pt';torch.save(model.state_dict(),path)
   choices.append(dict(resolution=res,seed=seed,epoch=best[2],selection_bce=best[0],threshold=cut,training_seconds=seconds,parameters=sum(p.numel() for p in model.parameters()),weights_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
   for condition,im in variants.items():
    p=expit(logits(model,torch.tensor(im[:,None],dtype=torch.float32)));pred[f'r{res}_s{seed}_{condition}']=p
    rows.append(dict(resolution=res,seed=seed,condition=condition,threshold=cut,**metrics(labels,p),**detection(labels,p,cut)))
   (OUT/f'progress_r{res}_s{seed}.json').write_text(json.dumps(choices[-1],indent=2))
 for name,records in [('results',rows),('learning_curves',history)]:
  with (OUT/f'{name}.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
 np.savez_compressed(OUT/'predictions.npz',labels=labels,test_indices=idx['test'],**pred)
 (OUT/'run.json').write_text(json.dumps(dict(config=config,choices=choices,torch=torch.__version__,elapsed_seconds=time.perf_counter()-started,commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),manifest_sha256=hashlib.sha256((ROOT/'configs/resolution_manifest.json').read_bytes()).hexdigest()),indent=2))
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser()
 parser.add_argument('--config',type=Path,default=ROOT/'configs/resolution_study.json')
 parser.add_argument('--output',type=Path,default=OUT)
 parser.add_argument('--models',type=Path,default=ROOT/'models/resolution')
 args=parser.parse_args();main(args.config,args.output,args.models)
