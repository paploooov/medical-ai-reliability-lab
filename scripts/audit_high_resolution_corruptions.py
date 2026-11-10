"""Measure realized uint8 corruption strength without using labels or predictions."""
from pathlib import Path
import csv
import hashlib
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
from high_resolution_robustness import transform,VIEWS


def main():
    cfg=json.loads((ROOT/'configs/high_resolution_robustness.json').read_text())
    indices=json.loads((ROOT/'configs/resolution_manifest.json').read_text())['indices']['test']
    images=np.load(ROOT/'data/pneumoniamnist_224.npz')['test_images'][indices]
    rows=[]
    for view in VIEWS:
        for condition in cfg['conditions'][1:]:
            rmse=[]
            for image,index in zip(images,indices):
                clean=transform(image,index,'clean',view).astype(float)/255
                corrupted=transform(image,index,condition,view).astype(float)/255
                rmse.append(float(np.sqrt(np.mean((corrupted-clean)**2))))
            rows.append(dict(view=view,condition=condition,n=len(rmse),mean_rmse=float(np.mean(rmse)),
                             minimum_rmse=min(rmse),maximum_rmse=max(rmse)))
    out=ROOT/'reports/high_resolution/robustness'
    with (out/'corruption_strength.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
