"""Separate bilinear original-encoder cache, protocol c44117a."""
import importlib.metadata
import io
import json
import os
import platform
import resource
import time
import zipfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from prepare_original_xray import digest
from thoraxis_stage3_common import guard
from reproduce_512 import normalize

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT/'data/thoraxis_source_shift/bilinear'
MANIFEST = ROOT/'configs/thoraxis_bilinear_cache.json'
SOURCE = ROOT/'reports/original_512/manifest.json'
ARCHIVE = ROOT/'data/original_xray/ChestXRay2017.zip'
WEIGHTS = ROOT/'data/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth'

def letterbox(image,size):
    image=image.convert('L');w,h=image.size;scale=size/max(w,h)
    resized=image.resize((max(1,round(w*scale)),max(1,round(h*scale))),Image.Resampling.BILINEAR)
    out=Image.new('L',(size,size),0);out.paste(resized,((size-resized.width)//2,(size-resized.height)//2))
    return np.asarray(out)

def records():
    return json.loads(SOURCE.read_text())['records']

def identity():
    return {'source_sha256':digest(SOURCE),'archive_sha256':digest(ARCHIVE),
            'encoder_weights_sha256':digest(WEIGHTS),
            'ordered_image_ids_sha256':__import__('hashlib').sha256(json.dumps([r['name'] for r in records()],separators=(',',':')).encode()).hexdigest(),
            'preprocessing_code_sha256':digest(Path(__file__)),
            'normalization_code_sha256':digest(ROOT/'scripts/reproduce_512.py'),
            'cache_code_sha256':digest(Path(__file__)),
            'size':512,'kernel':'bilinear','dtype':'float32','python':platform.python_version(),
            'versions':{p:importlib.metadata.version(p) for p in ('numpy','torch','torchvision','pillow','scipy','scikit-learn')}}

def encoder():
    os.environ['TORCH_HOME']=str(ROOT/'data/torch')
    model=mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1).eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model

def verify():
    m=json.loads(MANIFEST.read_text())
    assert identity()==m['identity'], 'Cache identity changed'
    arrays={}
    for name,entry in m['files'].items():
        path=CACHE/name
        assert digest(path)==entry['sha256'], f'Corrupt cache: {name}'
        arr=np.load(path,mmap_mode='r')
        assert list(arr.shape)==entry['shape'] and str(arr.dtype)=='float32'
        arrays[name]=arr
    return m,arrays

def main():
    guard()
    start=time.perf_counter(); torch.set_num_threads(1)
    CACHE.mkdir(parents=True,exist_ok=False)
    ident=identity()
    assert ident['archive_sha256']==json.loads(SOURCE.read_text())['archive_sha256']
    rr=records(); model=encoder(); prefix=model.features[:10]
    tail=torch.nn.Sequential(model.features[10:],torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten())
    pooled=np.lib.format.open_memmap(CACHE/'pooled.npy',mode='w+',dtype='float32',shape=(len(rr),576))
    with zipfile.ZipFile(ARCHIVE) as zf, torch.inference_mode():
        for first in range(0,len(rr),16):
            batch=np.stack([letterbox(Image.open(io.BytesIO(zf.read(r['name']))),512) for r in rr[first:first+16]])
            z=prefix(normalize(batch)); f=tail(z)
            if first==0: cache=np.lib.format.open_memmap(CACHE/'prefix.npy',mode='w+',dtype='float32',shape=(len(rr),*z.shape[1:]))
            cache[first:first+len(batch)]=z.numpy(); pooled[first:first+len(batch)]=f.numpy()
            if first%512==0: print('Frozen features',first,len(rr),round(time.perf_counter()-start),flush=True)
        cache.flush(); pooled.flush(); checks=[]
        for part in ('train','selection','calibration'):
            ids=sorted((i for i,r in enumerate(rr) if r['split']==part),key=lambda i:rr[i]['name'])[:2]
            for i in ids:
                image=letterbox(Image.open(io.BytesIO(zf.read(rr[i]['name']))),512)
                direct=prefix(normalize(image[None])); full=tail(direct)
                np.testing.assert_allclose(cache[i],direct[0].numpy(),atol=2e-4,rtol=1e-4)
                np.testing.assert_allclose(pooled[i],full[0].numpy(),atol=2e-4,rtol=1e-4)
                checks.append({'image_id':rr[i]['name'],'partition':part,
                               'prefix_max_abs_error':float(np.max(np.abs(cache[i]-direct[0].numpy()))),
                               'pooled_max_abs_error':float(np.max(np.abs(pooled[i]-full[0].numpy())))})
    files={name:{'sha256':digest(CACHE/name),'shape':list(arr.shape),'bytes':(CACHE/name).stat().st_size,
                 'row_ids':'records order in source manifest, verified by identity hash'}
           for name,arr in [('prefix.npy',cache),('pooled.npy',pooled)]}
    m={'identity':ident,'files':files,'direct_checks':checks,'medical_test_predictions':0,
       'total_seconds':time.perf_counter()-start,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    MANIFEST.write_text(json.dumps(m,indent=2)+'\n'); verify()
    print('Cache verified',m['total_seconds'],m['peak_rss_mib'],flush=True)

if __name__=='__main__': main()
