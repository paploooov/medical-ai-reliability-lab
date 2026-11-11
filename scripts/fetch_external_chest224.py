"""Fetch official ChestMNIST224 with atomic completion and checksum validation."""
import hashlib
import json
from pathlib import Path
import urllib.request
ROOT=Path(__file__).resolve().parents[1]
def valid(path,cfg):
    if not path.exists() or path.stat().st_size!=cfg['archive_bytes']:return False
    h=hashlib.md5()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()==cfg['archive_md5']
def main():
    cfg=json.loads((ROOT/'configs/external_chest224.json').read_text());target=ROOT/'data/external_chest224/chestmnist_224.npz'
    if valid(target,cfg):print('Existing official archive verified');return
    if target.exists():raise RuntimeError('Existing archive is incomplete or invalid; preserve it and investigate before replacing')
    target.parent.mkdir(parents=True,exist_ok=True);partial=target.with_suffix('.npz.partial')
    with urllib.request.urlopen(cfg['source'],timeout=120) as response,partial.open('xb') as f:
        total=0
        while chunk:=response.read(1024*1024):
            total+=len(chunk)
            if total>cfg['archive_bytes']:raise RuntimeError('Response exceeds official archive size')
            f.write(chunk)
    if not valid(partial,cfg):raise RuntimeError('Archive size or MD5 mismatch; partial preserved for inspection')
    partial.rename(target);print('Official archive downloaded and verified')
if __name__=='__main__':main()
