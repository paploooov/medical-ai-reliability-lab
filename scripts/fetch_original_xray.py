"""Download the original chest archive only, with authoritative SHA-256."""
import json,urllib.request
from pathlib import Path
from prepare_original_xray import digest
ROOT=Path(__file__).resolve().parents[1]

def main():
    cfg=json.loads((ROOT/'configs/original_512.json').read_text());target=ROOT/'data/original_xray/ChestXRay2017.zip';target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if digest(target)!=cfg['archive_sha256']:raise ValueError('Existing archive checksum mismatch; preserve and inspect it')
        print('Verified existing original archive');return
    partial=target.with_suffix('.zip.partial')
    with urllib.request.urlopen(cfg['archive_url'],timeout=120) as source,partial.open('wb') as out:
        while chunk:=source.read(1024*1024):out.write(chunk)
    if digest(partial)!=cfg['archive_sha256']:raise ValueError('Downloaded archive checksum mismatch')
    partial.replace(target);print('Downloaded and verified',target)

if __name__=='__main__':main()
