"""Prepare a local checkpoint package; never uploads or grants a weights license."""
import hashlib,json,shutil,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    source=ROOT/'data/original_xray/mobile512_research.pt';expected='ba497699914926742ee07592297f803925e835c0e2491e2c284eb008804f01ce';assert sha(source)==expected
    out=ROOT/'data/release-staging/THORAXIS-512-R1';out.mkdir(parents=True,exist_ok=False)
    for source,name in [(source,'THORAXIS-512-R1.pt'),(ROOT/'CITATION.cff','CITATION.cff'),(ROOT/'docs/THORAXIS_MODEL_CARD.md','MODEL_CARD.md'),(ROOT/'docs/reproduction-512/RELEASE_REVIEW.md','RELEASE_REVIEW.md'),(ROOT/'docs/reproduction-512/TORCHVISION-LICENSE.txt','TORCHVISION-LICENSE.txt')]:shutil.copyfile(source,out/name)
    (out/'NOTICE.txt').write_text('THORAXIS 512-R1 local release candidate. NOT UPLOADED.\nWeights redistribution review remains open; no weights license is granted here.\nBase architecture/weights: Torchvision MobileNetV3-Small IMAGENET1K_V1.\nFine-tuning data: Kermany, Zhang & Goldbaum (2018), doi:10.17632/rscbjbr9sj.2, CC BY 4.0.\nChanges: grayscale, Lanczos letterbox512, linear classifier and partial encoder fine-tuning.\nCode citation/license does not license weights. Repository-relative model-card links refer to:\nhttps://github.com/paploooov/medical-ai-reliability-lab\n')
    manifest=dict(status='prepared_locally_no_upload',weights_license_status='unresolved',files={p.name:sha(p) for p in out.iterdir()})
    (out/'SHA256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    bundle=out.with_suffix('.zip')
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):z.write(p,p.name)
    report=dict(**manifest,bundle=str(bundle.relative_to(ROOT)),bundle_sha256=sha(bundle),bundle_bytes=bundle.stat().st_size)
    (ROOT/'docs/reproduction-512/release_manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
