"""Verify official archive, audit identities/pixels, and cache paired resolutions."""
import hashlib, io, json, re, zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def patient_id(name):
    stem=Path(name).stem
    m=re.match(r'(person\d+)_',stem,re.I)
    if m:return m.group(1).lower()
    m=re.match(r'((?:NORMAL2-)?IM-\d+)-',stem,re.I)
    if m:return m.group(1).lower()
    raise ValueError(f'Unknown patient naming convention: {name}')

def letterbox(image,size):
    image=image.convert('L');w,h=image.size;scale=size/max(w,h)
    resized=image.resize((max(1,round(w*scale)),max(1,round(h*scale))),Image.Resampling.LANCZOS)
    out=Image.new('L',(size,size),0)
    out.paste(resized,((size-resized.width)//2,(size-resized.height)//2))
    return np.asarray(out)

def main():
    cfg=json.loads((ROOT/'configs/original_512.json').read_text());cache=ROOT/'data/original_xray'
    report=ROOT/'reports/original_512';report.mkdir(exist_ok=True,parents=True)
    archive=cache/'ChestXRay2017.zip';assert digest(archive)==cfg['archive_sha256']
    z=zipfile.ZipFile(archive);records=[]
    names=sorted(n for n in z.namelist() if n.lower().endswith(('.jpeg','.jpg','.png')) and '/__MACOSX/' not in '/'+n and not Path(n).name.startswith('._'))
    for i,name in enumerate(names):
        parts=Path(name).parts
        split=next((s for s in ['train','test','val'] if s in parts),None)
        if split not in ['train','test']:continue
        label=int('PNEUMONIA' in parts)
        if 'NORMAL' not in parts and not label:raise ValueError(name)
        im=Image.open(io.BytesIO(z.read(name))).convert('L');a=np.asarray(im)
        pixel=hashlib.sha256(str(im.size).encode()+a.tobytes()).hexdigest()
        records.append(dict(name=name,original_split=split,label=label,group=patient_id(name),width=im.width,height=im.height,pixel_sha256=pixel))
        if i%1000==0:print('Audit',i,len(names),flush=True)
    seen={};retained=[];excluded=[]
    for r in sorted(records,key=lambda r:(r['original_split']!='test',r['name'])):
        h=r['pixel_sha256']
        if h in seen:
            if r['label']!=seen[h]['label']:raise ValueError('Conflicting labels for duplicate pixels')
            excluded.append(dict(name=r['name'],reason='exact_pixel_duplicate',kept=seen[h]['name']));continue
        seen[h]=r;retained.append(r)
    tg={r['group'] for r in retained if r['original_split']=='test'};clean=[]
    for r in retained:
        if r['original_split']=='train' and r['group'] in tg:excluded.append(dict(name=r['name'],reason='group_in_test'))
        else:clean.append(r)
    train=[r for r in clean if r['original_split']=='train'];labels=np.array([r['label'] for r in train]);groups=np.array([r['group'] for r in train])
    folds=StratifiedGroupKFold(10,shuffle=True,random_state=cfg['seed'])
    for fold,(_,indices) in enumerate(folds.split(np.zeros(len(train)),labels,groups)):
        for i in indices:train[i]['split']={0:'selection',1:'calibration'}.get(fold,'train')
    for r in clean:
        if r['original_split']=='test':r['split']='test'
    clean=sorted(clean,key=lambda r:r['name'])
    summary={}
    for split in ['train','selection','calibration','test']:
        rows=[r for r in clean if r['split']==split]
        summary[split]=dict(n=len(rows),positive=sum(r['label'] for r in rows),groups=len({r['group'] for r in rows}),long_edge_below512=sum(max(r['width'],r['height'])<512 for r in rows),short_edge_below512=sum(min(r['width'],r['height'])<512 for r in rows),width_range=[min(r['width'] for r in rows),max(r['width'] for r in rows)],height_range=[min(r['height'] for r in rows),max(r['height'] for r in rows)])
    for a in summary:
        for b in summary:
            if a!=b:assert not ({r['group'] for r in clean if r['split']==a}&{r['group'] for r in clean if r['split']==b})
    manifest=dict(source=cfg['source'],archive_sha256=digest(archive),protocol_sha256=digest(ROOT/'configs/original_512.json'),preparation_sha256=digest(__file__),counts=summary,excluded=excluded,records=clean,scope='Filename-derived grouping, not independently verified clinical IDs. Exact decoded grayscale duplicates only.')
    (report/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(summary),flush=True)
    for size in [224,512]:
        path=cache/f'images_{size}.npy'
        arr=np.lib.format.open_memmap(path,mode='w+',dtype='uint8',shape=(len(clean),size,size))
        for i,r in enumerate(clean):
            arr[i]=letterbox(Image.open(io.BytesIO(z.read(r['name']))),size)
            if i%1000==0:print('Resize',size,i,len(clean),flush=True)
        arr.flush();del arr
    (cache/'cache_provenance.json').write_text(json.dumps({'manifest_sha256':digest(report/'manifest.json'),'images':{str(s):digest(cache/f'images_{s}.npy') for s in [224,512]}},indent=2)+'\n')

if __name__=='__main__':main()
