import io
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import open_clip
from tqdm import tqdm


IMAGES_DIR = "/data/yefengshuo.yfs/datasets/images"
BATCH_SIZE = 64
_IMAGE_SUFFIXES = frozenset(
    ".jpg .jpeg .png ".split()
)

# 模型路径与 compatibility2 中 UDF 一致
BLIP_MODEL_DIR = "/data/yefengshuo.yfs/download_src/pretrained_model/blip_caption_model"
OPEN_CLIP_MODEL_PATH = "/data/yefengshuo.yfs/download_src/pretrained_model/vit_b32_model/open_clip_model.safetensors"


def load_all_image_bytes(images_dir: str) -> np.ndarray:
    root = Path(images_dir)
    paths = sorted(
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    )
    n = len(paths)
    out = np.empty(n, dtype=object)
    for i, p in enumerate(paths):
        with open(p, "rb") as f:
            out[i] = f.read()
    return out


def bench_file_size() -> None:
    t0 = time.perf_counter()
    images = load_all_image_bytes(IMAGES_DIR)
    n = len(images)
    file_sizes = np.empty(n, dtype=np.int64)
    for i in range(n):
        file_sizes[i] = len(images[i])
    t1 = time.perf_counter()
    _ = file_sizes  # 丢弃结果，仅保留副作用防优化
    print(f"[file_size] 耗时: {t1 - t0:.3f} s")


def bench_dimension() -> None:
    t0 = time.perf_counter()
    images = load_all_image_bytes(IMAGES_DIR)
    n = len(images)
    dimensions = np.empty(n, dtype=object)
    for i in range(n):
        img = Image.open(io.BytesIO(images[i]))
        dimensions[i] = json.dumps({"width": img.size[0], "height": img.size[1]})
    t1 = time.perf_counter()
    _ = dimensions
    print(f"[dimension] 耗时: {t1 - t0:.3f} s")


def bench_caption() -> None:
    images = load_all_image_bytes(IMAGES_DIR)
    t0 = time.perf_counter()

    processor = BlipProcessor.from_pretrained(BLIP_MODEL_DIR, local_files_only=True)
    model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_DIR, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    n = len(images)
    out = np.empty(n, dtype=object)

    batch_starts = range(0, n, BATCH_SIZE)
    for start in tqdm(batch_starts, desc="caption", unit="batch"):
        end = min(start + BATCH_SIZE, n)
        chunk = images[start:end]
        pil_images = [Image.open(io.BytesIO(b)).convert("RGB") for b in chunk]
        with torch.no_grad():
            inputs = processor(pil_images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            gen = model.generate(**inputs, max_length=50)
            captions = processor.batch_decode(gen, skip_special_tokens=True)
        for i, cap in enumerate(captions):
            out[start + i] = cap

    t1 = time.perf_counter()
    _ = out
    print(f"[caption] 耗时: {t1 - t0:.3f} s")


def bench_embedding() -> None:
    images = load_all_image_bytes(IMAGES_DIR)
    t0 = time.perf_counter()

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained=OPEN_CLIP_MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    n = len(images)
    out = np.empty(n, dtype=object)

    batch_starts = range(0, n, BATCH_SIZE)
    for start in tqdm(batch_starts, desc="embedding", unit="batch"):
        end = min(start + BATCH_SIZE, n)
        chunk = images[start:end]
        pil_images = [Image.open(io.BytesIO(b)).convert("RGB") for b in chunk]
        with torch.no_grad():
            x = torch.stack([preprocess(img) for img in pil_images]).to(device)
            emb = model.encode_image(x)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            vecs = emb.cpu().float().numpy()
        for j in range(vecs.shape[0]):
            out[start + j] = vecs[j].astype(np.float32).tobytes()

    t1 = time.perf_counter()
    _ = out
    print(f"[embedding] 耗时: {t1 - t0:.3f} s")

if __name__ == "__main__":
    n = sum(
        1
        for p in Path(IMAGES_DIR).iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    )
    print("图片文件数:", n, "IMAGES_DIR:", IMAGES_DIR)

    bench_file_size()
    bench_dimension()
    bench_caption()
    bench_embedding()
