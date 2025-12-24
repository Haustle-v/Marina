import time
import pyarrow as pa

# Ensure we import local pyseekdb from this repo (./src) instead of any installed package.
import json
import os
import sys
from typing import Optional
from urllib.parse import urlparse

sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from pyseekdb.client import db


def _redact_uri(uri: str) -> str:
    """
    Redact sensitive parts of a connection URI to avoid leaking credentials/internal hosts
    into logs.
    """
    try:
        parsed = urlparse(uri)
        host = parsed.hostname or ""
        # Only allow localhost to be printed; everything else is treated as sensitive.
        safe_host = host if host in ("127.0.0.1", "localhost") else "<redacted-host>"
        userinfo = "<user>:<password>@" if (parsed.username is not None or parsed.password is not None) else ""
        port = f":{parsed.port}" if parsed.port is not None else ""
        # Keep path/query to help debugging without disclosing the actual host.
        return parsed._replace(netloc=f"{userinfo}{safe_host}{port}").geturl()
    except Exception:
        return "<redacted-uri>"


def _load_observer_uri() -> str:
    uri_file = os.path.join(os.getcwd(), ".pyseekdb_test_ob_uri")
    if os.path.exists(uri_file):
        with open(uri_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    return "mysql://root:@127.0.0.1:2881/test?tenant=mysql"


def _load_ray_address() -> Optional[str]:
    # Optional: if you want backfill to run on a remote Ray cluster, create a file
    # named ".pyseekdb_test_ray_address" in repo root with a single line:
    #   ray://<ray-head-host>:10001
    addr_file = os.path.join(os.getcwd(), ".pyseekdb_test_ray_address")
    if os.path.exists(addr_file):
        with open(addr_file, "r", encoding="utf-8") as f:
            v = f.read().strip()
            return v if v else None
    return None


def test_backfill_compatibility():
    """
    Integration-style test (same style as test_add_column_compat.py):
    - Connect to an OceanBase observer via mysql:// URI
    - Create a table with a primary key column `id`
    - Add the target backfill column
    - Run backfill to compute that column using a Python UDF executed in Ray
    - Verify results via SELECT

    How to run:
      1) Put observer URI into .pyseekdb_test_ob_uri (gitignored)
      2) (optional) Put ray address into .pyseekdb_test_ray_address (gitignored)
      3) python3.11 test_backfill_compat.py
    """
    # if sys.version_info < (3, 11):
    #     print("This script requires Python >= 3.11. Please run: python3.11 test_backfill_compat.py")
    #     return

    uri = _load_observer_uri()
    ray_address = _load_ray_address()
    ray_init_kwargs = {"namespace": "pyseekdb"}
    # Make sure Ray workers can import the local `src/pyseekdb` package when running locally.
    if ray_address is None:
        ray_init_kwargs["runtime_env"] = {
            "env_vars": {"PYTHONPATH": os.path.join(os.getcwd(), "src")},
        }

    print("Initializing PySeekDB connection (backfill compat)...")
    print("Connecting to", _redact_uri(uri))
    try:
        conn = db.connect(uri)
    except Exception as e:
        print("Connection failed, please update .pyseekdb_test_ob_uri:", e)
        return

    # Ensure ray is available (backfill_async requires it)
    try:
        import ray  # noqa: F401
    except Exception as e:
        print("Ray is required for backfill test. Please install ray. Error:", e)
        return

    table_name = "test_backfill_" + str(int(time.time()))
    print("Creating table:", table_name)

    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("a", pa.int64()),
            pa.field("text", pa.string()),
        ]
    )
    data = [
        {"id": 1, "a": 10, "text": "hello"},
        {"id": 2, "a": 7, "text": "oceanbase"},
        {"id": 3, "a": 0, "text": ""},
        {"id": 4, "a": 3, "text": "ray-offload"},
        {"id": 5, "a": -1, "text": "测试"},
        {"id": 6, "a": 99, "text": None},
    ]

    try:
        tbl = conn.create_table(table_name, data=data, schema=schema, mode="overwrite")
    except Exception as e:
        print("Create table failed, please ensure observer is running and uri is correct:", e)
        return

    print("Adding target column out...")
    tbl.add_column("out", pa.int64(), nullable=True)

    # RecordBatch UDF: input is a RecordBatch with column `a`, output is an Arrow array.
    def calc_out(batch: pa.RecordBatch) -> pa.Array:
        vals = batch.column(batch.schema.get_field_index("a")).to_pylist()
        out_vals = []
        for v in vals:
            if v is None:
                out_vals.append(None)
            else:
                out_vals.append(int(v) * 2)
        return pa.array(out_vals, type=pa.int64())

    print("Running backfill on Ray...")
    job_id = tbl.backfill(
        "out",
        udf=calc_out,
        read_columns=["a"],
        key_column="id",
        batch_size=2,
        ray_address=ray_address,
        # Example: if you want to isolate jobs in a namespace on remote ray
        ray_init_kwargs=ray_init_kwargs,
    )
    print("Backfill job_id:", job_id)

    print("Adding embedding columns emb_json and emb_pid...")
    # Use JSON column for embeddings to avoid vector type differences across deployments.
    tbl.add_column("emb_json", pa.list_(pa.float32()), nullable=True)
    tbl.add_column("emb_pid", pa.int64(), nullable=True)

    # Embedding UDF: deterministic 4-dim float embedding from text.
    # This UDF runs inside Ray worker processes via pyseekdb backfill.
    def calc_emb_json(batch: pa.RecordBatch) -> pa.Array:
        idx = batch.schema.get_field_index("text")
        texts = batch.column(idx).to_pylist()
        out_vals = []
        for t in texts:
            if t is None:
                out_vals.append(None)
            else:
                s = str(t)
                code_sum = 0
                for ch in s:
                    code_sum += ord(ch)
                vec = [
                    float(len(s)),
                    float(code_sum % 97),
                    float((code_sum // 97) % 97),
                    float(code_sum % 7),
                ]
                out_vals.append(vec)
        return pa.array(out_vals, type=pa.list_(pa.float32()))

    # PID UDF: write Ray worker PID into table to prove work is executed in Ray workers
    # (PID should differ from driver PID).
    def calc_emb_pid(batch: pa.RecordBatch) -> pa.Array:
        n = batch.num_rows
        pid = int(os.getpid())
        return pa.array([pid] * n, type=pa.int64())

    print("Running embedding backfill on Ray...")
    emb_job_id = tbl.backfill(
        "emb_json",
        udf=calc_emb_json,
        read_columns=["text"],
        key_column="id",
        batch_size=2,
        ray_address=ray_address,
        ray_init_kwargs=ray_init_kwargs,
    )
    print("Embedding backfill job_id:", emb_job_id)

    print("Running embedding PID backfill on Ray...")
    pid_job_id = tbl.backfill(
        "emb_pid",
        udf=calc_emb_pid,
        read_columns=["text"],
        key_column="id",
        batch_size=2,
        ray_address=ray_address,
        ray_init_kwargs=ray_init_kwargs,
    )
    print("Embedding PID backfill job_id:", pid_job_id)

    print("Query back and verify...")
    rows = conn._client_proxy._server._execute(
        f"SELECT `id`,`a`,`text`,`out`,`emb_json`,`emb_pid` FROM `{table_name}` ORDER BY `id`"
    )
    print("Rows:", rows)

    norm_rows = []
    for r in rows:
        if isinstance(r, dict):
            norm_rows.append(
                (r["id"], r["a"], r["text"], r["out"], r["emb_json"], r["emb_pid"])
            )
        else:
            norm_rows.append(tuple(r))

    expected_out = [
        (1, 10, "hello", 20),
        (2, 7, "oceanbase", 14),
        (3, 0, "", 0),
        (4, 3, "ray-offload", 6),
        (5, -1, "测试", -2),
        (6, 99, None, 198),
    ]
    for i, row in enumerate(norm_rows):
        rid, a, text, out, emb_json, emb_pid = row
        exp_id, exp_a, exp_text, exp_out = expected_out[i]
        if rid != exp_id or a != exp_a or text != exp_text or out != exp_out:
            raise RuntimeError(f"Unexpected base results: got={row}, expected={expected_out[i]}")

        # Verify embedding JSON is present and deterministic when text is not None.
        if text is None:
            if emb_json is not None:
                raise RuntimeError(f"Expected emb_json NULL for id={rid}, got={emb_json!r}")
        else:
            # MySQL/OB may return JSON as string; normalize to Python list.
            emb_val = emb_json
            if isinstance(emb_val, str):
                emb_val = json.loads(emb_val)
            if not isinstance(emb_val, list) or len(emb_val) != 4:
                raise RuntimeError(f"Invalid emb_json for id={rid}: {emb_json!r}")
            # Recompute expected embedding and compare with tolerance.
            s = str(text)
            code_sum = 0
            for ch in s:
                code_sum += ord(ch)
            exp_vec = [
                float(len(s)),
                float(code_sum % 97),
                float((code_sum // 97) % 97),
                float(code_sum % 7),
            ]
            for j in range(4):
                if abs(float(emb_val[j]) - float(exp_vec[j])) > 1e-6:
                    raise RuntimeError(
                        f"Unexpected emb_json for id={rid}: got={emb_json!r}, expected={exp_vec}"
                    )

        # Verify PID is from Ray worker process (should differ from driver PID).
        if emb_pid is None:
            raise RuntimeError(f"Expected emb_pid non-NULL for id={rid}")
        if int(emb_pid) == int(os.getpid()):
            raise RuntimeError(
                "emb_pid equals driver PID; expected backfill to run in Ray workers"
            )
    print("✅ backfill compatibility test passed!")


if __name__ == "__main__":
    test_backfill_compatibility()




