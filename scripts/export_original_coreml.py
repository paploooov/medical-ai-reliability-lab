"""Export a research-only 512px model; Apple runtime validation remains separate."""
import json,sys,zipfile
from pathlib import Path
import numpy as np
import torch
from torchvision.models import mobilenet_v3_small
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from prepare_original_xray import digest

class RawGrayscaleModel(torch.nn.Module):
    def __init__(self,model):
        super().__init__();self.model=model
        self.register_buffer('mean',torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('std',torch.tensor([.229,.224,.225])[None,:,None,None])
    def forward(self,pixels):
        return self.model((pixels.repeat(1,3,1,1)/255-self.mean)/self.std)

def main():
    import coremltools as ct
    torch.set_num_threads(1);cache=ROOT/'data/original_xray';out=cache/'deployment';out.mkdir(exist_ok=True)
    saved=torch.load(cache/'mobile512_research.pt',weights_only=True,map_location='cpu');model=mobilenet_v3_small(weights=None);model.classifier=torch.nn.Linear(576,1);model.load_state_dict(saved['state_dict']);wrapper=RawGrayscaleModel(model.eval()).eval()
    raw=np.broadcast_to(np.linspace(0,255,512,dtype=np.float32)[None,None,None,:],(1,1,512,512)).copy();x=torch.from_numpy(raw)
    with torch.inference_mode():reference=wrapper(x).numpy()
    traced=torch.jit.trace(wrapper,x);np.testing.assert_allclose(traced(x).detach().numpy(),reference,atol=1e-5)
    converted=ct.convert(traced,inputs=[ct.TensorType(name='pixels',shape=(1,1,512,512),dtype=np.float32)],outputs=[ct.TensorType(name='logit',dtype=np.float32)],minimum_deployment_target=ct.target.iOS16,compute_precision=ct.precision.FLOAT32,convert_to='mlprogram',skip_model_load=True)
    spec=converted.get_spec()
    assert spec.WhichOneof('Type')=='mlProgram'
    assert spec.description.input[0].name=='pixels'
    assert list(spec.description.input[0].type.multiArrayType.shape)==[1,1,512,512]
    assert spec.description.output[0].name=='logit'
    converted.short_description='Exploratory pediatric chest-image classifier. Not clinically validated.'
    converted.user_defined_metadata['preprocessing']='Grayscale, full image aspect-preserving letterbox512, black padding; raw pixels 0..255. Normalization is inside model.'
    converted.user_defined_metadata['scope']='Research only. No clinical deployment. Apple runtime numerical parity pending.'
    path=out/'Mobile512Research.mlpackage';converted.save(str(path))
    raw.astype('<f4').tofile(out/'gradient512.f32');fixture=dict(shape=[1,1,512,512],dtype='little-endian float32',input_file='gradient512.f32',input_sha256=digest(out/'gradient512.f32'),expected_logit=float(reference[0,0]),tolerance=1e-3)
    (out/'fixture.json').write_text(json.dumps(fixture,indent=2)+'\n')
    contract=dict(converted=True,coremltools=ct.__version__,torch=torch.__version__,minimum_ios=16,input_name='pixels',input_shape=[1,1,512,512],input_range=[0,255],output_name='logit',class_mapping={'0':'NORMAL','1':'PNEUMONIA'},probability='sigmoid(logit), not a validated clinical probability',calibration_cutoff=saved['cutoff'],selected_epoch=saved['selected_epoch'],checkpoint_sha256=digest(cache/'mobile512_research.pt'),torchscript_wrapper_parity=True,apple_runtime_parity=False,device_latency_measured=False,preprocessing_parity_on_apple=False,files={str(p.relative_to(out)):digest(p) for p in path.rglob('*') if p.is_file()},fixture=fixture)
    bundle=cache/'SwiftUIResearchBundle.zip'
    readme='''# SwiftUI research tensor probe

Create an iOS 16+ SwiftUI project in Xcode. Add TensorProbeView.swift and use
TensorProbeView() as its content view. Add Mobile512Research.mlpackage,
gradient512.f32 and fixture.json to the app target. Xcode compiles the model.
Run the reference tensor check first with CPU-only compute.

This bundle was generated under Linux. Xcode compilation, Apple numerical
parity, device latency and image preprocessing have NOT been validated.
The input is synthetic; this is not a diagnostic application.

Input: pixels, Float32 [1,1,512,512], grayscale values 0..255. Normalization is
inside the model. Output: logit for the dataset PNEUMONIA class. The sigmoid
is not a validated individual clinical probability. A future image pipeline
must reproduce orientation, grayscale conversion, aspect-preserving Lanczos
resize and black padding; the tensor probe does not test those steps.

Own source code: LICENSE. Dataset and model provenance/limitations:
DATA_AND_MODEL_LICENSES.md. No original patient images are included.
'''
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(path.rglob('*')):
            if f.is_file():z.write(f,str(f.relative_to(out)))
        for name in ['gradient512.f32','fixture.json']:z.write(out/name,name)
        z.write(ROOT/'ios/ResearchPrototype/TensorProbeView.swift','TensorProbeView.swift');z.write(ROOT/'LICENSE','LICENSE');z.write(ROOT/'docs/LICENSING.md','DATA_AND_MODEL_LICENSES.md');z.writestr('README.md',readme)
    contract['research_bundle']=str(bundle.relative_to(ROOT));contract['research_bundle_sha256']=digest(bundle);contract['coreml_schema_verified']=True;contract['coreml_specification_version']=spec.specificationVersion
    (ROOT/'reports/original_512/deployment.json').write_text(json.dumps(contract,indent=2)+'\n');print('Core ML package exported; Apple runtime testing still pending.',flush=True)

if __name__=='__main__':main()
