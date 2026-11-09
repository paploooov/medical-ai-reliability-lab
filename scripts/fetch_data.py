"""Fetch only the checksum-verified 28px archive, without rewriting research outputs."""
from pathlib import Path
import hashlib,json,tempfile,urllib.request
ROOT=Path(__file__).resolve().parents[1]

def main():
    metadata=json.loads((ROOT/'reports/dataset_summary.json').read_text())
    path=ROOT/'data/pneumoniamnist.npz';path.parent.mkdir(exist_ok=True)
    if path.exists():
        if hashlib.md5(path.read_bytes()).hexdigest()!=metadata['md5']:
            raise ValueError('Existing archive has an unexpected checksum; left untouched')
    else:
        with urllib.request.urlopen(metadata['source'],timeout=60) as response:
            content=response.read()
        if hashlib.md5(content).hexdigest()!=metadata['md5']:raise ValueError('Download checksum mismatch')
        with path.open('xb') as target:target.write(content)
    print('PneumoniaMNIST 28px: checksum verified')

if __name__=='__main__':main()
