"""Download the official 224px archive with a pinned checksum, without rewriting outputs."""
from pathlib import Path
import hashlib,json,urllib.request
ROOT=Path(__file__).resolve().parents[1]

def main():
    cfg=json.loads((ROOT/'configs/high_resolution_probe.json').read_text())
    target=ROOT/'data/pneumoniamnist_224.npz';target.parent.mkdir(exist_ok=True)
    if not target.exists():
        partial=target.with_suffix('.npz.part')
        with urllib.request.urlopen(cfg['source_url'],timeout=60) as source,partial.open('wb') as out:
            while chunk:=source.read(1024*1024):out.write(chunk)
        if hashlib.md5(partial.read_bytes()).hexdigest()!=cfg['source_md5']:
            raise ValueError('High-resolution archive checksum mismatch; partial not promoted')
        partial.rename(target)
    assert hashlib.md5(target.read_bytes()).hexdigest()==cfg['source_md5']
    print(f'Official 224px archive verified: {target.stat().st_size} bytes',flush=True)

if __name__=='__main__':main()
