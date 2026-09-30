"""Скачивает CIFAR-10 и раскладывает в train/val по классам."""
import os, pickle, tarfile, urllib.request, shutil
from pathlib import Path

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
ROOT = Path("/data/cifar10")
RAW = ROOT / "raw"

def download():
    RAW.mkdir(parents=True, exist_ok=True)
    if not (RAW / "cifar-10-batches-py").exists():
        tgz = RAW / "cifar-10-python.tar.gz"
        urllib.request.urlretrieve(URL, tgz)
        with tarfile.open(tgz) as t:
            t.extractall(RAW)

def unpack(split, files):
    out = ROOT / split
    out.mkdir(parents=True, exist_ok=True)
    for f in files:
        with open(RAW / "cifar-10-batches-py" / f, "rb") as fp:
            d = pickle.load(fp, encoding="latin1")
        for img, label in zip(d["data"], d["labels"]):
            cls_dir = out / str(label)
            cls_dir.mkdir(exist_ok=True)
            import numpy as np
            from PIL import Image
            arr = img.reshape(3, 32, 32).transpose(1, 2, 0)
            Image.fromarray(arr).save(cls_dir / f"{f}_{len(list(cls_dir.iterdir()))}.png")

if __name__ == "__main__":
    download()
    unpack("train", [f"data_batch_{i}" for i in range(1, 6)])
    unpack("val", ["test_batch"])
    print("✅ dataset ready at /data/cifar10")
