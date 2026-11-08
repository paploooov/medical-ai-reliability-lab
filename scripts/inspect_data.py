"""Official PneumoniaMNIST archive: validate splits and display training examples."""
from pathlib import Path
import hashlib
import json
import urllib.request
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
URL = "https://zenodo.org/records/10519652/files/pneumoniamnist.npz?download=1"
MD5 = "28209eda62fecd6e6a2d98b1501bb15f"

def main():
    path = ROOT / "data/pneumoniamnist.npz"
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        temp = path.with_suffix(".download")
        urllib.request.urlretrieve(URL, temp)
        if hashlib.md5(temp.read_bytes()).hexdigest() != MD5:
            temp.unlink()
            raise ValueError("Download checksum mismatch")
        temp.replace(path)
    if hashlib.md5(path.read_bytes()).hexdigest() != MD5:
        raise ValueError("Dataset checksum mismatch; remove the corrupt local file")
    summary = {"source": URL, "md5": MD5, "seed": 42, "splits": {}}
    with np.load(path, allow_pickle=False) as data:
        for split, count in {"train": 4708, "val": 524, "test": 624}.items():
            images, labels = data[f"{split}_images"], data[f"{split}_labels"]
            assert images.shape == (count, 28, 28)
            assert labels.shape == (count, 1)
            assert images.dtype == np.uint8
            assert set(np.unique(labels)) == {0, 1}
            summary["splits"][split] = {"count": count, "normal": int((labels == 0).sum()), "pneumonia": int((labels == 1).sum())}
        indices = np.random.default_rng(42).choice(4708, 20, replace=False)
        fig, axes = plt.subplots(4, 5, figsize=(10, 8))
        for ax, index in zip(axes.flat, indices):
            ax.imshow(data["train_images"][index], cmap="gray", vmin=0, vmax=255)
            label = int(data["train_labels"][index, 0])
            ax.set_title(f"#{index}: " + ["normal", "pneumonia"][label], fontsize=9)
            ax.axis("off")
        fig.suptitle("PneumoniaMNIST: 20 training examples (dataset labels)")
        fig.tight_layout()
        out = ROOT / "reports"
        out.mkdir(exist_ok=True)
        fig.savefig(out / "training_examples.png", dpi=140)
        plt.close(fig)
        summary["displayed_training_indices"] = indices.tolist()
        (out / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
