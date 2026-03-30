import io
import random
import shutil
from pathlib import Path
import time

import pyarrow as pa
import pyarrow.compute as pc
from PIL import Image
import geneva
from geneva.tqdm import tqdm

from geneva import udf
import numpy as np
import open_clip
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor

OPEN_CLIP_MODEL_PATH = "/data/yefengshuo.yfs/download_src/pretrained_model/vit_b32_model/open_clip_model.safetensors"
BLIP_MODEL_DIR = "/data/yefengshuo.yfs/download_src/pretrained_model/blip_caption_model"

# IMAGES_DIR = "/data/1/projects/marina/datasets/images_demo"
IMAGES_DIR = "/data/yefengshuo.yfs/datasets/images"

GENEVA_DB_PATH = "/data/1/projects/marina/database_geneva"
NUM_IMAGES = None  # 只取前 N 张；设为 None 表示全部


def _iter_image_paths():
    paths = sorted(
        p
        for p in Path(IMAGES_DIR).iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if NUM_IMAGES is not None:
        paths = paths[:NUM_IMAGES]
    return paths


def load_images(frag_size: int = 25):
    paths = _iter_image_paths()
    batch = []
    for idx, path in enumerate(tqdm(paths)):
        buf = io.BytesIO()
        with Image.open(path) as im:
            im.convert("RGB").save(buf, format="png")
        batch.append(
            {
                "image": buf.getvalue(),
                "image_id": idx,
            }
        )
        if len(batch) >= frag_size:
            yield pa.RecordBatch.from_pylist(batch)
            batch = []
    if batch:
        yield pa.RecordBatch.from_pylist(batch)


def create_table(overwrite: bool = True):
    if not overwrite:
        db = geneva.connect(GENEVA_DB_PATH)
        return db, db.open_table("images")

    shutil.rmtree(GENEVA_DB_PATH, ignore_errors=True)
    db = geneva.connect(GENEVA_DB_PATH)

    first = True
    tbl = None
    for batch in load_images():
        if first:
            tbl = db.create_table("images", batch, mode="overwrite")
            first = False
        else:
            tbl.add(batch)
    return db, tbl

@udf()
def file_size(image: bytes) -> int:
    return len(image)

@udf(data_type=pa.struct([
    pa.field("width", pa.int32()),
    pa.field("height", pa.int32())]))
def dimensions(image: bytes):
    img = Image.open(io.BytesIO(image))
    return {"width": img.size[0], "height": img.size[1]}

@udf(cuda=True)
class BlipCaptioner:
    def __init__(self): self.is_loaded = False
    def setup(self):
        self.processor = BlipProcessor.from_pretrained(
            str(BLIP_MODEL_DIR), local_files_only=True
        )
        self.model = BlipForConditionalGeneration.from_pretrained(
            str(BLIP_MODEL_DIR), local_files_only=True
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device); self.is_loaded = True
    def __call__(self, image: bytes) -> str:
        if not self.is_loaded: self.setup()
        raw = Image.open(io.BytesIO(image)).convert("RGB")
        inputs = self.processor([raw], return_tensors="pt")
        inputs = {k: v.to(self.device) for k,v in inputs.items()}
        out = self.model.generate(**inputs, max_length=50)
        return self.processor.decode(out[0], skip_special_tokens=True)



@udf(cuda=True, data_type=pa.list_(pa.float32(), 512))
class GenEmbeddings:
    def __init__(self):
        self.is_loaded = False

    def setup(self):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained=str(OPEN_CLIP_MODEL_PATH)
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True

    def __call__(self, image: bytes):
        if not self.is_loaded:
            self.setup()
        raw = Image.open(io.BytesIO(image)).convert("RGB")
        x = self.preprocess(raw).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_image(x)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        vec = emb.cpu().float().numpy().flatten()
        return vec.astype(np.float32).tolist()


def _cast_binary_to_large_binary(table: pa.Table) -> pa.Table:
    """Avoid PyArrow take() offset overflow on wide binary columns (e.g. image bytes)."""
    cols = []
    for name in table.column_names:
        col = table[name]
        if pa.types.is_binary(col.type):
            cols.append(pc.cast(col, pa.large_binary()))
        else:
            cols.append(col)
    return pa.table(cols, names=table.column_names)


def print_table_preview(tbl, n: int = 2, *, seed: int | None = None) -> None:
    print("schema:", tbl.schema)
    at = tbl.to_arrow()
    total = at.num_rows
    k = min(n, total)
    if k == 0:
        print("(0 rows)")
        return
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(total), k))
    at = _cast_binary_to_large_binary(at)
    print(at.take(pa.array(indices, type=pa.int64())).to_pandas())


if __name__ == "__main__":
    db, tbl = create_table(overwrite=False)

    tbl.add_columns({"file_size": file_size, "dimensions": dimensions})
    tbl.add_columns({"caption": BlipCaptioner(), "embedding": GenEmbeddings()})

    with db.local_ray_context():
        t0 = time.perf_counter()
        tbl.backfill("file_size")
        t1 = time.perf_counter()
        tbl.backfill("dimensions")
        t2 = time.perf_counter()
        tbl.backfill("caption")
        t3 = time.perf_counter()
        tbl.backfill("embedding")
        t4 = time.perf_counter()
        print(f"file_size backfill time cost: {t1 - t0:.3f} seconds")
        print(f"dimensions backfill time cost: {t2 - t1:.3f} seconds")
        print(f"caption backfill time cost: {t3 - t2:.3f} seconds")
        print(f"embedding backfill time cost: {t4 - t3:.3f} seconds")

    print_table_preview(tbl, n=50, seed=42)
