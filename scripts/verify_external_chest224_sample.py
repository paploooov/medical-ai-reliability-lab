"""Reconstruct pixel exclusions and seeded sampling from the source archive."""
import hashlib
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
    folder=ROOT/'reports/external_chest224';cfg=json.loads((ROOT/'configs/external_chest224.json').read_text());archive=ROOT/'data/external_chest224/chestmnist_224.npz'
    h=hashlib.md5()
    with archive.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    assert h.hexdigest()==cfg['archive_md5'];assert archive.stat().st_size==cfg['archive_bytes']
    data=np.load(archive);images=data['test_images'];labels=data['test_labels'];groups={}
    for i,image in enumerate(images):groups.setdefault(hashlib.sha256(image.tobytes()).hexdigest(),[]).append(i)
    retained=[];conflicts=[];duplicates=[]
    for key,indices in groups.items():
        duplicates.extend([[i,indices[0]] for i in indices[1:]])
        if len(np.unique(labels[indices],axis=0))>1:conflicts.append(key)
        else:retained.append(indices[0])
    retained=np.array(sorted(retained));expected=json.loads((folder/'sampling.json').read_text())
    assert sorted(duplicates)==sorted(expected['duplicate_pairs']);assert sorted(conflicts)==expected['conflicting_pixel_hashes'];assert len(retained)==expected['retained_after_dedup']
    positives=retained[labels[retained,6]==1];normal=retained[labels[retained].sum(1)==0];other=retained[(labels[retained,6]==0)&(labels[retained].sum(1)>0)];rng=np.random.default_rng(20261127)
    chosen=np.r_[positives,np.sort(rng.choice(normal,min(2000,len(normal)),replace=False)),np.sort(rng.choice(other,min(1000,len(other)),replace=False))]
    saved=np.load(folder/'sample.npz');np.testing.assert_array_equal(chosen,saved['indices']);np.testing.assert_array_equal(labels[chosen],saved['original_labels'])
    assert expected['positive_with_additional_tags']==int((labels[positives].sum(1)>1).sum())
    report=dict(passed=True,archive_md5=h.hexdigest(),all_test_images_checked=len(images),sample_indices_reconstructed=True,selected_count=len(chosen),conflicting_hash_groups=len(conflicts),duplicate_pairs=len(duplicates))
    (folder/'sampling_verification.json').write_text(json.dumps(report,indent=2)+'\n');print('Official archive, exclusions and deterministic sample verified')
if __name__=='__main__':main()
