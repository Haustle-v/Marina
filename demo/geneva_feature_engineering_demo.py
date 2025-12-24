#!/usr/bin/env python3
"""
Geneva feature engineering demo (LanceDB + Ray), distilled from:
https://lancedb.com/blog/geneva-feature-engineering/

This script mirrors the blog flow:
  1) Install/check environment (LanceDB, Ray, datasets, Pillow)
  2) Load Oxford-IIIT Pets dataset into a local LanceDB database via Geneva
  3) Define UDFs for:
     - file_size (scalar)
     - dimensions (struct)
     - captions (optional: BLIP; fallback "light" caption)
     - embeddings (optional: OpenCLIP; fallback "light" embedding)
  4) Run backfills on a Ray cluster (local Ray by default)
  5) Query results

Notes:
  - By default this runs in "light" mode (no heavy model deps) but still uses Ray.
  - Use --use-blip / --use-openclip if you have those dependencies installed.
"""

import argparse
import io
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path
from typing import Iterator, Optional

import pyarrow as pa


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _setup_sys_path() -> None:
    # Use the vendored Geneva code in this repo (geneva-0.7.0/src)
    root = _repo_root()
    geneva_src = root / "geneva-0.7.0" / "src"
    sys.path.insert(0, str(geneva_src))


def _load_batches(num_images: int, frag_size: int) -> Iterator[pa.RecordBatch]:
    # Import heavy deps lazily so "pip install" errors are clearer.
    from datasets import load_dataset

    def _yield_from_rows(rows) -> Iterator[pa.RecordBatch]:
        batch = []
        for row in rows:
            buf = io.BytesIO()
            row["image"].save(buf, format="png")
            batch.append(
                {
                    "image": buf.getvalue(),
                    "label": int(row["label"]),
                    "image_id": str(row["image_id"]),
                    "label_cat_dog": int(row["label_cat_dog"]),
                }
            )
            if len(batch) >= frag_size:
                yield pa.RecordBatch.from_pylist(batch)
                batch = []
        if batch:
            yield pa.RecordBatch.from_pylist(batch)

    try:
        dataset = load_dataset(
            "timm/oxford-iiit-pet",
            split=f"train[:{int(num_images)}]",
        )
        yield from _yield_from_rows(dataset)
        return
    except Exception as e:
        # Many CI/air-gapped environments cannot reach HuggingFace.
        # Fall back to generating deterministic synthetic images so the Ray + LanceDB
        # execution path is still testable end-to-end.
        print("WARN: failed to download Oxford Pets from HuggingFace; using synthetic data.", e)

    from PIL import Image, ImageDraw

    rows = []
    for i in range(int(num_images)):
        w = 128 + (i % 4) * 16
        h = 128 + ((i // 4) % 4) * 16
        img = Image.new("RGB", (w, h), color=((i * 37) % 255, (i * 71) % 255, (i * 13) % 255))
        draw = ImageDraw.Draw(img)
        draw.rectangle([(10, 10), (w - 10, h - 10)], outline=(255, 255, 255), width=3)
        draw.text((14, 14), f"id={i}", fill=(255, 255, 0))
        label_cat_dog = 1 if (i % 2 == 0) else 0
        rows.append(
            {
                "image": img,
                "label": i % 37,
                "image_id": f"synthetic_{i}",
                "label_cat_dog": label_cat_dog,
            }
        )
    yield from _yield_from_rows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Geneva feature engineering demo (LanceDB + Ray)")
    parser.add_argument("--db-path", default=str(_repo_root() / "geneva_demo_db"), help="Local DB dir")
    parser.add_argument("--table", default="images", help="Table name")
    parser.add_argument("--num-images", type=int, default=12, help="Number of images to load")
    parser.add_argument("--frag-size", type=int, default=4, help="RecordBatch size when ingesting")
    parser.add_argument("--batch-size", type=int, default=8, help="Backfill read batch size")
    parser.add_argument("--concurrency", type=int, default=2, help="Ray actor concurrency")
    parser.add_argument("--commit-granularity", type=int, default=2, help="Partial commit granularity")
    parser.add_argument("--overwrite", action="store_true", help="Delete db dir before running")

    parser.add_argument(
        "--oxford-images-tgz",
        default=str(_repo_root() / "images.tar.gz"),
        help="Local Oxford Pets images.tar.gz (preferred when present)",
    )
    parser.add_argument(
        "--oxford-annotations-tgz",
        default=str(_repo_root() / "annotations.tar.gz"),
        help="Local Oxford Pets annotations.tar.gz (preferred when present)",
    )
    parser.add_argument(
        "--prefer-local-oxford",
        action="store_true",
        help="Prefer local Oxford tarballs even if HuggingFace is reachable",
    )

    parser.add_argument(
        "--ray-address",
        default=None,
        help="Optional Ray address (e.g. ray://<head>:10001). If unset, uses local Ray.",
    )

    parser.add_argument("--use-blip", action="store_true", help="Use BLIP caption UDF (requires deps)")
    parser.add_argument("--use-openclip", action="store_true", help="Use OpenCLIP embedding UDF (requires deps)")
    return parser.parse_args()


def _load_oxford_from_local_archives(
    images_tgz: Path,
    annotations_tgz: Path,
    num_images: int,
    frag_size: int,
) -> Iterator[pa.RecordBatch]:
    """
    Load Oxford-IIIT Pets from local archives:
      - images.tar.gz contains: images/<name>.jpg
      - annotations.tar.gz contains: annotations/trainval.txt

    trainval.txt format (space separated):
      <image_id> <class_id> <species_id> <breed_id>
    species_id: 1=cat, 2=dog
    """
    if not images_tgz.exists():
        raise FileNotFoundError(f"images tarball not found: {images_tgz}")
    if not annotations_tgz.exists():
        raise FileNotFoundError(f"annotations tarball not found: {annotations_tgz}")
    if num_images <= 0:
        raise ValueError("num_images must be > 0")
    if frag_size <= 0:
        raise ValueError("frag_size must be > 0")

    with tarfile.open(annotations_tgz, "r:gz") as ann_tf:
        member = ann_tf.getmember("annotations/trainval.txt")
        f = ann_tf.extractfile(member)
        if f is None:
            raise RuntimeError("Failed to read annotations/trainval.txt from annotations tarball")
        lines = f.read().decode("utf-8", errors="replace").splitlines()

    items = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        image_id = parts[0]
        class_id = int(parts[1])
        species_id = int(parts[2])
        items.append((image_id, class_id, species_id))
        if len(items) >= int(num_images):
            break

    if not items:
        raise RuntimeError("No items parsed from annotations/trainval.txt")

    # Build a fast lookup for image tar members we need.
    wanted = set()
    for image_id, _, _ in items:
        wanted.add(f"images/{image_id}.jpg")

    with tarfile.open(images_tgz, "r:gz") as img_tf:
        members = {}
        for m in img_tf.getmembers():
            if m.name in wanted:
                members[m.name] = m
        # Materialize rows
        batch = []
        for image_id, class_id, species_id in items:
            path = f"images/{image_id}.jpg"
            m = members.get(path)
            if m is None:
                raise RuntimeError(f"Image not found in images tarball: {path}")
            f = img_tf.extractfile(m)
            if f is None:
                raise RuntimeError(f"Failed to read image bytes: {path}")
            image_bytes = f.read()

            label = int(class_id) - 1  # normalize to 0-based like HF datasets commonly do
            label_cat_dog = 1 if int(species_id) == 2 else 0
            batch.append(
                {
                    "image": image_bytes,
                    "label": label,
                    "image_id": str(image_id),
                    "label_cat_dog": label_cat_dog,
                }
            )
            if len(batch) >= int(frag_size):
                yield pa.RecordBatch.from_pylist(batch)
                batch = []
        if batch:
            yield pa.RecordBatch.from_pylist(batch)


def main() -> int:
    _setup_sys_path()

    # Imports after sys.path setup
    import geneva
    from geneva.cluster import GenevaClusterType
    from geneva.runners.ray._mgr import ray_cluster

    args = _parse_args()

    db_path = Path(args.db_path).expanduser().absolute()
    if args.overwrite and db_path.exists():
        shutil.rmtree(db_path)
    db_path.mkdir(parents=True, exist_ok=True)

    print("DB path:", db_path)
    print("Loading dataset...")

    images_tgz = Path(args.oxford_images_tgz).expanduser().absolute()
    annotations_tgz = Path(args.oxford_annotations_tgz).expanduser().absolute()
    use_local = args.prefer_local_oxford or (images_tgz.exists() and annotations_tgz.exists())

    if use_local:
        print("Using local Oxford Pets archives:")
        print("  images:", images_tgz)
        print("  annotations:", annotations_tgz)
        batches = list(
            _load_oxford_from_local_archives(
                images_tgz, annotations_tgz, args.num_images, args.frag_size
            )
        )
    else:
        batches = list(_load_batches(args.num_images, args.frag_size))
    if not batches:
        raise RuntimeError("No batches loaded")

    print(f"Loaded {args.num_images} rows in {len(batches)} batches")

    conn = geneva.connect(db_path)

    print("Creating table...")
    t_write_start = time.perf_counter()
    tbl = conn.create_table(args.table, batches[0], mode="overwrite")
    for b in batches[1:]:
        tbl.add(b)
    write_secs = time.perf_counter() - t_write_start
    print(f"TIMING write_lancedb_secs={write_secs:.3f}")

    # --------------------
    # Define UDFs
    # --------------------
    from geneva import udf

    @udf
    def file_size(image: bytes) -> int:
        return len(image)

    @udf(
        data_type=pa.struct(
            [
                pa.field("width", pa.int32()),
                pa.field("height", pa.int32()),
            ]
        )
    )
    def dimensions(image: bytes):
        from PIL import Image

        img = Image.open(io.BytesIO(image))
        return {"width": int(img.size[0]), "height": int(img.size[1])}

    # Captions
    caption_udf = None
    if args.use_blip:
        try:
            import torch
            from PIL import Image
            from transformers import BlipForConditionalGeneration, BlipProcessor
        except Exception as e:
            raise RuntimeError(
                "BLIP requested but dependencies are missing. Install: "
                "pip install transformers torch accelerate"
            ) from e

        @udf
        class BlipCaptioner:
            def __init__(self) -> None:
                self._loaded = False
                self._processor = None
                self._model = None
                self._device = None

            def _setup(self) -> None:
                self._processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self._model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self._model.to(self._device)
                self._loaded = True

            def __call__(self, image: bytes) -> str:
                if not self._loaded:
                    self._setup()
                raw = Image.open(io.BytesIO(image)).convert("RGB")
                inputs = self._processor([raw], return_tensors="pt")
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                out = self._model.generate(**inputs, max_length=50)
                return self._processor.decode(out[0], skip_special_tokens=True)

        caption_udf = BlipCaptioner()
    else:
        # Light caption: deterministic and dependency-light, still exercises Ray execution.
        @udf
        def light_caption(label: int, label_cat_dog: int) -> str:
            kind = "dog" if int(label_cat_dog) == 1 else "cat"
            return f"pet={kind}, label={int(label)}"

        caption_udf = (light_caption, ["label", "label_cat_dog"])

    # Embeddings
    embedding_udf = None
    if args.use_openclip:
        try:
            import numpy as np
            import open_clip
            from PIL import Image
            import torch
        except Exception as e:
            raise RuntimeError(
                "OpenCLIP requested but dependencies are missing. Install: "
                "pip install open-clip-torch torch"
            ) from e

        @udf(data_type=pa.list_(pa.float32(), 512))
        class OpenClipEmbeddings:
            def __init__(self) -> None:
                self._loaded = False
                self._model = None
                self._preprocess = None
                self._device = None

            def _setup(self) -> None:
                model, _, preprocess = open_clip.create_model_and_transforms(
                    "ViT-B-32", pretrained="laion2b_s34b_b79k"
                )
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(self._device)
                model.eval()
                self._model = model
                self._preprocess = preprocess
                self._loaded = True

            def __call__(self, image: bytes):
                if not self._loaded:
                    self._setup()
                raw = Image.open(io.BytesIO(image)).convert("RGB")
                x = self._preprocess(raw).unsqueeze(0).to(self._device)
                with torch.no_grad():
                    emb = self._model.encode_image(x)
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                vec = emb.squeeze(0).detach().cpu().numpy().astype("float32")
                return vec.tolist()

        embedding_udf = OpenClipEmbeddings()
    else:
        # Light embedding: stable float vector derived from image bytes.
        @udf(data_type=pa.list_(pa.float32(), 32))
        def light_embedding(image: bytes) -> pa.Array:
            # Simple hash-based embedding; deterministic and cheap.
            n = 32
            acc = [0] * n
            for i, b in enumerate(image[: 4096]):
                acc[i % n] = (acc[i % n] + int(b)) % 997
            vec = [float(v) / 997.0 for v in acc]
            return pa.array(vec, type=pa.float32())

        embedding_udf = light_embedding

    # Create UDF columns
    tbl.add_columns(
        {
            "file_size": file_size,
            "dimensions": dimensions,
            "caption": caption_udf,
            "embedding": embedding_udf,
        }
    )

    # --------------------
    # Run backfills on Ray
    # --------------------
    def _run_backfill(col: str, *, async_mode: bool) -> None:
        t0 = time.perf_counter()
        if async_mode:
            fut = tbl.backfill_async(
                col,
                batch_size=args.batch_size,
                concurrency=args.concurrency,
                commit_granularity=args.commit_granularity,
            )
            while not fut.done(timeout=5):
                tbl.checkout_latest()
                try:
                    done = int(tbl.count_rows(filter=f"{col} is not null"))
                except Exception:
                    done = 0
                print(f"committed {done} rows for {col}")
            fut.result()
        else:
            tbl.backfill(
                col,
                batch_size=args.batch_size,
                concurrency=args.concurrency,
                commit_granularity=args.commit_granularity,
            )

        tbl.checkout_latest()
        dt = time.perf_counter() - t0
        if col == "embedding":
            print(f"TIMING embedding_backfill_secs={dt:.3f}")
        else:
            print(f"TIMING backfill_{col}_secs={dt:.3f}")

    print("Starting Ray and running backfills...")

    ray_ctx = None
    if args.ray_address:
        # Attach to an existing Ray cluster.
        ray_ctx = ray_cluster(
            addr=args.ray_address,
            use_portforwarding=False,
            log_to_driver=True,
            ray_init_kwargs={"include_dashboard": True},
        )
    else:
        # Start a local Ray cluster. Geneva's progress/status utilities rely on the
        # Ray dashboard state API (default port 8265), so keep dashboard enabled.
        ray_ctx = ray_cluster(
            local=True,
            log_to_driver=True,
            ray_init_kwargs={"include_dashboard": True},
        )

    with ray_ctx:
        # Light features can run sync; heavier ones default to async for partial commits.
        _run_backfill("file_size", async_mode=False)
        _run_backfill("dimensions", async_mode=False)
        _run_backfill("caption", async_mode=True)
        _run_backfill("embedding", async_mode=True)

    print("Querying a few rows...")
    out = tbl.search().limit(5).to_arrow()
    print(out.to_pandas())

    print("✅ Demo completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


