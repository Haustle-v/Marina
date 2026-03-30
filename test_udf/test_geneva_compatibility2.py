import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import open_clip
from tqdm import tqdm


sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from pyseekdb.client import db



def _load_observer_uri() -> str:
    uri_file = os.path.join(os.getcwd(), ".pyseekdb_test_ob_uri")
    if os.path.exists(uri_file):
        with open(uri_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    return "mysql://root:@127.0.0.1:2881/test?tenant=mysql"


def create_image_paths_table(images_dir: Path):
    uri = _load_observer_uri()

    conn = db.connect(uri)
    print("Connected to observer:", uri)

    rows = [{"filename": path.name} for path in sorted(images_dir.iterdir())]
    print("Loaded", len(rows), "image paths")

    table_name = "test_lancedb_compat_" + str(int(time.time()))
    schema = pa.schema([
        pa.field("filename", pa.string()),
    ])
    tbl = conn.create_table(table_name, data=None, schema=schema, mode="overwrite")
    insert_batch_size = 200
    for i in tqdm(
        range(0, len(rows), insert_batch_size),
        desc="Insert rows",
    ):
        chunk = rows[i : i + insert_batch_size]
        tbl.add(chunk)
    print("Table", table_name, "created successfully with", len(rows), "rows")

    return tbl


def load_images_from_filenames(filenames: np.ndarray[str]) -> np.ndarray[bytes]:
    # worker节点的images存放目录
    IMAGES_DIR = "/data/yefengshuo.yfs/datasets/images"
    n = len(filenames)
    out = np.empty(n, dtype=object)
    for i in range(n):
        path = os.path.join(IMAGES_DIR, str(filenames[i]))
        with open(path, "rb") as f:
            out[i] = f.read()
    return out


def add_column_file_size(tbl):
    def file_size_udf(filenames: np.ndarray[str]) -> np.ndarray[int]:
        images = load_images_from_filenames(filenames)
        n = len(images)
        file_sizes = np.empty(n, dtype=np.int64)
        for i in range(n):
            file_sizes[i] = len(images[i])
        return file_sizes

    tbl.add_column("file_size", pa.int64(), udf=file_size_udf, udf_name="file_size_udf", input_columns=["filename"])
    print("Start to backfill file_size column")
    t0 = time.perf_counter()
    tbl.backfill("file_size")
    print(f"file_size column backfilled successfully, time cost: {time.perf_counter() - t0:.3f} seconds")


def add_column_dimension(tbl):
    def dimensions_udf(filenames: np.ndarray[str]) -> np.ndarray[str]:
        images = load_images_from_filenames(filenames)
        n = len(images)
        dimensions = np.empty(n, dtype=np.object_)
        for i in range(n):
            img = Image.open(io.BytesIO(images[i]))
            dimensions[i] = json.dumps({"width": img.size[0], "height": img.size[1]})
        return dimensions

    tbl.add_column("dimension", pa.string(), udf=dimensions_udf, udf_name="dimensions_udf", input_columns=["filename"])
    print("Start to backfill dimension column")
    t0 = time.perf_counter()
    tbl.backfill("dimension")
    print(f"dimension column backfilled successfully, time cost: {time.perf_counter() - t0:.3f} seconds")

def add_column_caption(tbl):
    def generate_caption_udf(filenames: np.ndarray[str]) -> np.ndarray[str]:
        images = load_images_from_filenames(filenames)
        # 如果受限于网络，可以直接从 https://huggingface.co/Salesforce/blip-image-captioning-base/tree/main 下载模型，本地加载
        # processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        # model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        # model_dir = "/data/1/projects/marina/pretrained_model/blip_caption_model"
        model_dir = "/data/yefengshuo.yfs/download_src/pretrained_model/blip_caption_model"
        processor = BlipProcessor.from_pretrained(model_dir, local_files_only=True)
        model = BlipForConditionalGeneration.from_pretrained(model_dir, local_files_only=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        n = len(images)
        out = np.empty(n, dtype=np.object_)
        for i in range(n):
            raw = Image.open(io.BytesIO(images[i])).convert("RGB")
            inputs = processor([raw], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            gen = model.generate(**inputs, max_length=50)
            out[i] = processor.decode(gen[0], skip_special_tokens=True)
        return out

    def generate_caption_udf_batch(filenames: np.ndarray[str]) -> np.ndarray[str]:
        images = load_images_from_filenames(filenames)
        # 如果受限于网络，可以直接从 https://huggingface.co/Salesforce/blip-image-captioning-base/tree/main 下载模型，本地加载
        # processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        # model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        # model_dir = "/data/1/projects/marina/pretrained_model/blip_caption_model"
        # 如果要用gpu，卸载到另一台机器上去计算，那么这里的路径就得写另一台机器存放模型的路径
        model_dir = "/data/yefengshuo.yfs/download_src/pretrained_model/blip_caption_model"
        processor = BlipProcessor.from_pretrained(model_dir, local_files_only=True)
        model = BlipForConditionalGeneration.from_pretrained(model_dir, local_files_only=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        n = len(images)
        out = np.empty(n, dtype=np.object_)
        pil_images = [Image.open(io.BytesIO(b)).convert("RGB") for b in images]
        with torch.no_grad():
            inputs = processor(pil_images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            gen = model.generate(**inputs, max_length=50)
            captions = processor.batch_decode(gen, skip_special_tokens=True)
        for i, cap in enumerate(captions):
            out[i] = cap
        return out

    tbl.add_column("caption", pa.string(), udf=generate_caption_udf, udf_name="generate_caption_udf", input_columns=["filename"])
    print("Start to backfill caption column")
    t0 = time.perf_counter()
    tbl.backfill("caption", num_gpus=4, num_batches=1)
    print(f"caption column backfilled successfully, time cost: {time.perf_counter() - t0:.3f} seconds")


def add_column_embedding(tbl):
    # preprocess → forward pass → normalize → return 512-d vector
    def gen_embedding_udf(filenames: np.ndarray[str]) -> np.ndarray[bytes]:
        images = load_images_from_filenames(filenames)
        # model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
        # 如果受限于网络，可以直接从 https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K/tree/main 下载模型，本地加载
        # model_path = "/data/1/projects/marina/pretrained_model/vit_b32_model/open_clip_model.safetensors"
        # 如果要用gpu，卸载到另一台机器上去计算，那么这里的路径就得写另一台机器存放模型的路径
        model_path = "/data/yefengshuo.yfs/download_src/pretrained_model/vit_b32_model/open_clip_model.safetensors"
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained=model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        n = len(images)
        out = np.empty(n, dtype=np.object_)
        for i in range(n):
            raw = Image.open(io.BytesIO(images[i])).convert("RGB")
            x = preprocess(raw).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model.encode_image(x)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            vec = emb.cpu().float().numpy().flatten()
            out[i] = vec.astype(np.float32).tobytes()
        return out

    # preprocess → forward pass → normalize → return 512-d vector
    def gen_embedding_udf_batch(filenames: np.ndarray[str]) -> np.ndarray[bytes]:
        images = load_images_from_filenames(filenames)
        # model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
        # 如果受限于网络，可以直接从 https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K/tree/main 下载模型，本地加载
        # model_path = "/data/1/projects/marina/pretrained_model/vit_b32_model/open_clip_model.safetensors"
        # 如果要用gpu，卸载到另一台机器上去计算，那么这里的路径就得写另一台机器存放模型的路径
        model_path = "/data/yefengshuo.yfs/download_src/pretrained_model/vit_b32_model/open_clip_model.safetensors"
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained=model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        n = len(images)
        out = np.empty(n, dtype=np.object_)
        pil_images = [Image.open(io.BytesIO(images[i])).convert("RGB") for i in range(n)]
        with torch.no_grad():
            # (N, C, H, W)
            x = torch.stack([preprocess(img) for img in pil_images]).to(device)
            emb = model.encode_image(x)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            vecs = emb.cpu().float().numpy()
        for i in range(n):
            out[i] = vecs[i].astype(np.float32).tobytes()
        return out

    tbl.add_column("embedding", pa.binary(), udf=gen_embedding_udf, udf_name="gen_embedding_udf", input_columns=["filename"])
    print("Start to backfill embedding column")
    t0 = time.perf_counter()
    tbl.backfill("embedding", num_gpus=4, num_batches=1)
    print(f"embedding column backfilled successfully, time cost: {time.perf_counter() - t0:.3f} seconds")


# 上面的IMAGES_DIR是worker节点的images存放目录，model_dir是模型存放目录，按需修改
if __name__ == "__main__":
    tbl = create_image_paths_table(Path("/data/1/projects/marina/datasets/images"))

    add_column_file_size(tbl)
    add_column_dimension(tbl)
    add_column_caption(tbl)
    add_column_embedding(tbl)

