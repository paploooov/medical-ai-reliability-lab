"""Synthetic golden images/tensors for a future Apple preprocessing parity test."""
import json,zipfile,sys
from pathlib import Path
import numpy as np
from PIL import Image,__version__ as pillow_version
import torch
from torchvision.models import mobilenet_v3_small
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from original_512 import normalize
from prepare_original_xray import letterbox,digest

def main():
    torch.set_num_threads(1);cache=ROOT/'data/original_xray';out=cache/'preprocessing_fixtures';out.mkdir(exist_ok=False)
    checkpoint=cache/'mobile512_research.pt';saved=torch.load(checkpoint,weights_only=True);model=mobilenet_v3_small(weights=None);model.classifier=torch.nn.Linear(576,1);model.load_state_dict(saved['state_dict']);model.eval();cases=[]
    for name,w,h,mode in [('gray_tie_down',1024,641,'L'),('gray_tie_up',1024,643,'L'),('rgb_tall',319,701,'RGB'),('small_gray',173,91,'L')]:
        yy,xx=np.indices((h,w));a=((3*xx+5*yy)%256).astype(np.uint8)
        if mode=='RGB':a=np.stack([xx%256,yy%256,(xx+yy)%256],axis=-1).astype(np.uint8)
        if name=='small_gray':a=(((xx//7+yy//5)%2)*255).astype(np.uint8)
        image=Image.fromarray(a);case=out/name;case.mkdir();image.save(case/'source.png');expected=letterbox(image,512);Image.fromarray(expected).save(case/'expected512.png');expected.astype('<f4')[None,None].tofile(case/'gradient512.f32')
        with torch.inference_mode():logit=float(model(normalize(expected[None])).item())
        scale=512/max(w,h);rw=max(1,round(w*scale));rh=max(1,round(h*scale));fixture=dict(shape=[1,1,512,512],dtype='little-endian float32',input_file='gradient512.f32',input_sha256=digest(case/'gradient512.f32'),expected_logit=logit,tolerance=1e-3)
        (case/'fixture.json').write_text(json.dumps(fixture,indent=2)+'\n');cases.append(dict(name=name,source_size=[w,h],source_mode=mode,resized_size=[rw,rh],padding_left_top=[(512-rw)//2,(512-rh)//2],files={p.name:digest(p) for p in case.iterdir()},expected_logit=logit))
        np.testing.assert_array_equal(np.fromfile(case/'gradient512.f32',dtype='<f4').reshape(512,512),np.asarray(Image.open(case/'expected512.png')))
    readme='''# Synthetic preprocessing parity fixtures

No medical images are included. Cases cover odd aspect ratios, small-image
upsampling, RGB-to-grayscale conversion and round-to-nearest-even geometry.
1024x641 must resize to512x320; 1024x643 must resize to512x322.

Reference: Pillow L conversion, LANCZOS full-image aspect-preserving resize,
round-to-nearest-even dimensions, centered black padding. Orientation is already
upright; these fixtures do not test EXIF orientation, DICOM windowing or camera input.

For an Apple image pipeline, compare its output pixels against expected512.png
and gradient512.f32. The strict reference contract is exact pixel equality.
Any different interpolation/color implementation needs separately justified
validation; no acceptable pixel-error budget has been clinically established.

The existing TensorProbeView can test each case's tensor inference: copy that
case's gradient512.f32 and fixture.json into the app bundle alongside the existing
Mobile512Research model. This checks model inference only, not image decoding
or resizing. The historical tensor filename is retained for compatibility.

Reference logits come from the frozen PyTorch model. CoreML execution, Xcode
compilation and image preprocessing on Apple hardware remain unverified.
'''
    (out/'README.md').write_text(readme);bundle=cache/'PreprocessingParityFixtures.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,str(p.relative_to(out)))
    report=dict(pillow=pillow_version,checkpoint_sha256=digest(checkpoint),cases=cases,pixel_tensor_roundtrip_verified=True,synthetic_only=True,apple_image_pipeline_verified=False,bundle=str(bundle.relative_to(ROOT)),bundle_sha256=digest(bundle))
    (ROOT/'reports/original_512/preprocessing/fixtures.json').write_text(json.dumps(report,indent=2)+'\n');print('Four synthetic preprocessing cases saved')

if __name__=='__main__':main()
