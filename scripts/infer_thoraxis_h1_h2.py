"""One frozen test/calibration forward pass per published model; no fitting."""
import io,json,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from torchvision.models import mobilenet_v3_small
from prepare_thoraxis_cache import verify,encoder,records as source_records
from prepare_original_xray import digest
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/original_512/thoraxis_30'

def main():
    torch.set_num_threads(1)
    receipt=json.loads((OUT/'pretest_publication.json').read_text())
    remote=subprocess.check_output(['git','ls-remote','origin','main'],cwd=ROOT,text=True).strip().split()[0]
    subprocess.run(['git','merge-base','--is-ancestor',receipt['checkpoint_commit'],remote],cwd=ROOT,check=True)
    state=json.loads((ROOT/'data/thoraxis_source_shift/training_batch/state.json').read_text())
    assert len(state['runs'])==30 and all(v['status']=='completed' for v in state['runs'].values())
    published=json.loads((OUT/'checkpoints/manifest.json').read_text())
    assert len(published['checkpoints'])==30
    for item in published['checkpoints']:
        assert digest(OUT/'checkpoints'/item['file'])==item['sha256']
    sentinel=OUT/'inference_started.json'
    with sentinel.open('x') as f:json.dump({'utc':datetime.now(timezone.utc).isoformat(),'remote_main':remote,'script_sha256':digest(__file__)},f,indent=2)
    _,arrays=verify();prefix=arrays['prefix.npy'];source=source_records();row={r['name']:i for i,r in enumerate(source)}
    initial=encoder().state_dict();plan=json.loads((ROOT/'configs/thoraxis_training_batch.json').read_text());results=[]
    for run in plan['runs']:
        started=time.monotonic();name=run['run_id'];path=OUT/'checkpoints'/f'{name}.pt'
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        manifest=ROOT/run['manifest'];assert digest(manifest)==checkpoint['manifest_sha256']
        assert checkpoint['seed']==run['seed'] and checkpoint['run_id']==name
        for k,v in initial.items():
            if k.startswith('features.') and int(k.split('.')[1])<10:assert torch.equal(v,checkpoint['state_dict'][k]),k
        model=mobilenet_v3_small(weights=None);model.classifier=torch.nn.Linear(576,1)
        model.load_state_dict(checkpoint['state_dict'],strict=True);model.eval()
        tail=torch.nn.Sequential(model.features[10:],torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten(),model.classifier).eval()
        rr=[r for r in json.loads(manifest.read_text())['records'] if r['split'] in ('test','calibration')]
        assert sum(r['split']=='test' for r in rr)==618
        ids=np.array([row[r['name']] for r in rr]);zz=[]
        with torch.inference_mode():
            for first in range(0,len(ids),32):zz.append(tail(torch.tensor(prefix[ids[first:first+32]].copy())).ravel().numpy())
        output=OUT/f'predictions_{name}.npz'
        with output.open('xb') as f:
            np.savez_compressed(f,logit=np.concatenate(zz),label=np.array([r['label'] for r in rr]),
                image_id=np.array([r['name'] for r in rr]),group=np.array([r['group'] for r in rr]),
                origin=np.array([r['original_split'] for r in rr]),partition=np.array([r['split'] for r in rr]),
                cutoff=np.float64(checkpoint['cutoff']))
        results.append(dict(run_id=name,file=output.name,sha256=digest(output),checkpoint_sha256=digest(path),
                            split_sha256=digest(manifest),cutoff=checkpoint['cutoff'],seconds=time.monotonic()-started))
        (OUT/'prediction_manifest.json').write_text(json.dumps({'models':results,'completed':len(results)==30,'inference_started':json.loads(sentinel.read_text())},indent=2)+'\n')
        print(name,'saved',len(results),'/30',flush=True)
if __name__=='__main__':main()
