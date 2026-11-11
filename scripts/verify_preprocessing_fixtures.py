"""Verify synthetic bundle hashes, reference geometry and pixel/tensor equality."""
import hashlib
import io
import json
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
def main():
    folder=ROOT/'reports/original_512/preprocessing';report=json.loads((folder/'fixtures.json').read_text());bundle=ROOT/report['bundle']
    assert hashlib.sha256(bundle.read_bytes()).hexdigest()==report['bundle_sha256']
    with zipfile.ZipFile(bundle) as archive:
        assert archive.testzip() is None
        for case in report['cases']:
            files={k:archive.read(case['name']+'/'+k) for k in case['files']}
            for name,expected in case['files'].items():assert hashlib.sha256(files[name]).hexdigest()==expected
            source=Image.open(io.BytesIO(files['source.png']));assert list(source.size)==case['source_size'];assert source.mode==case['source_mode']
            w,h=source.size;ratio=512/max(w,h);size=(round(w*ratio),round(h*ratio));assert list(size)==case['resized_size']
            expected=Image.new('L',(512,512),0);expected.paste(source.convert('L').resize(size,Image.Resampling.LANCZOS),tuple(case['padding_left_top']))
            png=np.asarray(Image.open(io.BytesIO(files['expected512.png'])));tensor=np.frombuffer(files['gradient512.f32'],dtype='<f4').reshape(512,512)
            np.testing.assert_array_equal(png,np.asarray(expected));np.testing.assert_array_equal(tensor,png)
            fixture=json.loads(files['fixture.json']);assert fixture['expected_logit']==case['expected_logit'];assert fixture['shape']==[1,1,512,512]
    (folder/'fixture_verification.json').write_text(json.dumps(dict(passed=True,cases=4,zip_crc=True,all_file_hashes=True,pixel_pipeline_reconstructed=True,apple_runtime_checked=False),indent=2)+'\n');print('Four synthetic pixel pipelines and all bundle hashes verified')
if __name__=='__main__':main()
