"""External test data: fixed RSNA Pneumonia sample, checksum-verified download and
PneumoniaMNIST-style preprocessing (grayscale, centre crop to short edge, resize).

Run: python experiments/external_data.py
Writes configs/external_rsna_sample.csv (committed) and data/external/rsna_{28,64}.npz (not committed).
"""
from pathlib import Path
import csv,hashlib,json,urllib.request,time
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
REPO='Baldezo313/rsna-pneumonia-dataset'
REVISION='a7f1dad69625cb4d02b9ef8ee849d5c0e6fc24b4'
BASE=f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/'
SAMPLE_SEED=20261109
DESIGN={'Lung Opacity':(1000,1,'primary'),'Normal':(1000,0,'primary'),'No Lung Opacity / Not Normal':(500,0,'secondary')}
SAMPLE_CSV=ROOT/'configs/external_rsna_sample.csv'
RAW=ROOT/'data/external/rsna'

def git_blob_sha1(content:bytes)->str:
 """Hash used by the Hub for non-LFS files (git blob object id)."""
 return hashlib.sha1(b'blob %d\0'%len(content)+content).hexdigest()

def to_uint8(pixels,photometric='MONOCHROME2'):
 """Grayscale uint8; inverts MONOCHROME1, min-max scales non-8-bit data."""
 a=np.asarray(pixels)
 if a.ndim!=2:raise ValueError('2D grayscale image required')
 if a.dtype!=np.uint8:
  a=a.astype(np.float64);lo,hi=a.min(),a.max()
  a=np.zeros_like(a) if hi==lo else (a-lo)/(hi-lo)*255
  a=np.round(a).astype(np.uint8)
 if photometric=='MONOCHROME1':a=255-a
 return a

def centre_crop(a):
 h,w=a.shape;s=min(h,w);top=(h-s)//2;left=(w-s)//2
 return a[top:top+s,left:left+s]

def preprocess(pixels,size,photometric='MONOCHROME2'):
 """MedMNIST description: centre crop with the short edge, then resize (PIL bicubic assumed)."""
 a=centre_crop(to_uint8(pixels,photometric))
 return np.asarray(Image.fromarray(a,mode='L').resize((size,size),resample=Image.Resampling.BICUBIC),dtype=np.uint8)

def draw_sample(class_rows,seed=SAMPLE_SEED,design=DESIGN):
 """class_rows: iterable of (patientId, class). One image per patientId; deterministic."""
 by_class={}
 for pid,c in class_rows:by_class.setdefault(c,set()).add(pid)
 rng=np.random.default_rng(seed);out=[]
 for c,(n,label,arm) in design.items():
  ids=sorted(by_class[c])
  if len(ids)<n:raise ValueError(f'not enough cases for {c}')
  for pid in sorted(rng.choice(ids,n,replace=False)):out.append(dict(patientId=str(pid),rsna_class=c,label=label,arm=arm))
 if len({r['patientId'] for r in out})!=len(out):raise ValueError('duplicate patientId')
 return out

def fetch(url,tries=4):
 for i in range(tries):
  try:
   with urllib.request.urlopen(url,timeout=60) as r:return r.read()
  except Exception:
   if i==tries-1:raise
   time.sleep(2*(i+1))

def paths_info(paths):
 req=urllib.request.Request(f'https://huggingface.co/api/datasets/{REPO}/paths-info/{REVISION}',data=json.dumps({'paths':paths}).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=60) as r:return {x['path']:x for x in json.load(r)}

def main():
 import pydicom
 from concurrent.futures import ThreadPoolExecutor
 RAW.mkdir(parents=True,exist_ok=True)
 info=json.loads(fetch(f'https://huggingface.co/api/datasets/{REPO}/revision/{REVISION}'))
 location={Path(s['rfilename']).stem:s['rfilename'] for s in info['siblings'] if s['rfilename'].startswith('stage_2_train_images')}
 for name in ['stage_2_detailed_class_info.csv','stage_2_train_labels.csv']:
  if not (RAW/name).exists():(RAW/name).write_bytes(fetch(BASE+name))
 classes=[(r['patientId'],r['class']) for r in csv.DictReader((RAW/'stage_2_detailed_class_info.csv').open())]
 targets={}
 for r in csv.DictReader((RAW/'stage_2_train_labels.csv').open()):targets.setdefault(r['patientId'],set()).add(int(r['Target']))
 rows=draw_sample(classes)
 for r in rows:
  assert targets[r['patientId']]=={r['label']},r
  r['path']=location[r['patientId']]
 meta={}
 for i in range(0,len(rows),500):meta.update(paths_info([r['path'] for r in rows[i:i+500]]))
 dicom=RAW/'dicom';dicom.mkdir(exist_ok=True)
 def get(r):
  target=dicom/(r['patientId']+'.dcm');expected=meta[r['path']]['oid']
  if target.exists() and git_blob_sha1(target.read_bytes())==expected:return
  content=fetch(BASE+r['path'])
  if git_blob_sha1(content)!=expected:raise ValueError('checksum mismatch '+r['path'])
  target.write_bytes(content)
 with ThreadPoolExecutor(8) as pool:list(pool.map(get,rows))
 images={28:[],64:[]}
 for r in rows:
  d=pydicom.dcmread(dicom/(r['patientId']+'.dcm'));px=d.pixel_array
  r.update(git_blob_sha1=meta[r['path']]['oid'],rows=int(d.Rows),columns=int(d.Columns),bits_stored=int(d.BitsStored),photometric=str(d.PhotometricInterpretation),view=str(d.get('ViewPosition','')),age=str(d.get('PatientAge','')),sex=str(d.get('PatientSex','')))
  for s in images:images[s].append(preprocess(px,s,r['photometric']))
 fields=['patientId','rsna_class','label','arm','path','git_blob_sha1','rows','columns','bits_stored','photometric','view','age','sex']
 with SAMPLE_CSV.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 for s,ims in images.items():
  np.savez_compressed(ROOT/f'data/external/rsna_{s}.npz',images=np.stack(ims),labels=np.array([r['label'] for r in rows],dtype=np.int64),arm=np.array([r['arm'] for r in rows]),patient_ids=np.array([r['patientId'] for r in rows]))
 print({a:sum(r['arm']==a for r in rows) for a in ['primary','secondary']},'sample csv sha256',hashlib.sha256(SAMPLE_CSV.read_bytes()).hexdigest())
if __name__=='__main__':main()
