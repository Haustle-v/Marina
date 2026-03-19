import time
import pyarrow as pa
import numpy as np

# Ensure we import local pyseekdb from this repo (./src) instead of any installed package.
import os
import sys

# sys.path.insert(0, os.path.join(os.getcwd(), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from pyseekdb.client import db


def _load_observer_uri() -> str:
    # Prefer a local, gitignored file to avoid committing sensitive info.
    # Create a file named ".pyseekdb_test_ob_uri" in repo root, with a single line:
    #   mysql://user:pass@host:port/db?tenant=xxx
    uri_file = os.path.join(os.getcwd(), ".pyseekdb_test_ob_uri")
    if os.path.exists(uri_file):
        with open(uri_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    # Fallback placeholder (edit the local file for real env)
    return "mysql://root:@127.0.0.1:2881/test?tenant=mysql"


def test_backfill_bigint():
    uri = _load_observer_uri()

    print("Initializing PySeekDB connection (test_backfill)...")
    print("Connecting to", uri)
    try:
        conn = db.connect(uri)
    except Exception as e:
        print("Connection failed, please update uri in pyseekdb_test_ob_uri:", e)
        return

    table_name = "test_backfill_bigint_" + str(int(time.time()))
    print("Creating table:", table_name)

    schema = pa.schema(
        [
            pa.field("a", pa.int64()),
            pa.field("b", pa.int64()),
            pa.field("c", pa.int64()),
        ]
    )

    # data = [{"a": 1, "b": 11, "c": 111}, {"a": 2, "b": 22, "c": 222}, {"a": 3, "b": 33, "c": 333}]
    data = [{"a": i, "b": 1000, "c": 10000} for i in range(0, 200)]
    
    tbl = conn.create_table(table_name, data=data, schema=schema, mode="overwrite")

    # Define Python UDFs: sum and product
    def add_udf(c1: np.ndarray, c2: np.ndarray, c3: np.ndarray) -> np.ndarray:    
        return c1 + c2 + c3
    print("Adding UDF generated column abc_sum = a + b + c (STORED via UDF)...")

    tbl.add_column("abc_sum", pa.int64(), udf=add_udf, udf_name="add_udf", input_columns=["a", "b", "c"])

    print("Backfilling UDF...")
    tbl.backfill("abc_sum")


def test_backfill_double():
    uri = _load_observer_uri()

    conn = db.connect(uri)

    table_name = "test_backfill_double_" + str(int(time.time()))
    print("Creating table:", table_name)

    schema = pa.schema([pa.field("x", pa.float64())])
    data = [{"x": 1.0}, {"x": 2.5}, {"x": 3.14}]

    tbl = conn.create_table(table_name, data=data, schema=schema, mode="overwrite")

    def double_udf(col_x: np.ndarray) -> np.ndarray:
        return col_x * 2.0

    print("Adding UDF generated column y = double_udf(x) ...")
    tbl.add_column("y", pa.float64(), udf=double_udf, udf_name="double_udf", input_columns=["x"])

    print("Backfilling UDF...")
    tbl.backfill("y")
    print("test_backfill_double done.")


def test_backfill_bool():
    uri = _load_observer_uri()
    conn = db.connect(uri)

    table_name = "test_backfill_bool_" + str(int(time.time()))
    print("Creating table:", table_name)

    schema = pa.schema([pa.field("A", pa.bool_()), pa.field("B", pa.bool_())])
    data = [
        {"A": True, "B": True},
        {"A": True, "B": False},
        {"A": False, "B": True},
        {"A": False, "B": False},
    ]

    tbl = conn.create_table(table_name, data=data, schema=schema, mode="overwrite")

    def xor_udf(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.logical_xor(a, b)

    print("Adding UDF generated column xor_ab = xor_udf(A, B) ...")
    tbl.add_column("xor_ab", pa.bool_(), udf=xor_udf, udf_name="xor_udf", input_columns=["A", "B"])

    print("Backfilling UDF...")
    tbl.backfill("xor_ab")
    print("test_backfill_bool done.")


def test_backfill_text():
    uri = _load_observer_uri()
    conn = db.connect(uri)

    table_name = "test_backfill_text_" + str(int(time.time()))
    print("Creating table:", table_name)

    schema = pa.schema([pa.field("text1", pa.string()), pa.field("text2", pa.string())])
    data = [
        {"text1": "a", "text2": "1"},
        {"text1": "hello", "text2": "world"},
        {"text1": "He", "text2": "Su"},
    ]

    tbl = conn.create_table(table_name, data=data, schema=schema, mode="overwrite")

    def concat_udf(text1: np.ndarray[np.object_], text2: np.ndarray[np.object_]) -> np.ndarray[np.object_]:
        return np.array([a + " " + b for a, b in zip(text1, text2)], dtype=np.object_)

    print("Adding UDF generated column concat = concat_udf(text1, text2) ...")
    tbl.add_column("concat", pa.string(), udf=concat_udf, udf_name="concat_udf", input_columns=["text1", "text2"])

    print("Backfilling UDF...")
    tbl.backfill("concat")
    print("test_backfill_text done.")


def test_backfill_blob():
    uri = _load_observer_uri()
    conn = db.connect(uri)

    table_name = "test_backfill_blob_" + str(int(time.time()))
    print("Creating table:", table_name)

    schema = pa.schema([pa.field("blob1", pa.binary()), pa.field("blob2", pa.binary())])
    data = [
        {"blob1": b"\x01\x02", "blob2": b"\x03"},
        {"blob1": b"ab", "blob2": b"cd"},
        {"blob1": b"x", "blob2": b"y"},
    ]

    tbl = conn.create_table(table_name, data=data, schema=schema, mode="overwrite")

    def concat_udf(blob1: np.ndarray[np.object_], blob2: np.ndarray[np.object_]) -> np.ndarray[np.object_]:
        return np.array([a + b for a, b in zip(blob1, blob2)], dtype=np.object_)

    print("Adding UDF generated column concat = concat_udf(blob1, blob2) ...")
    tbl.add_column("concat", pa.binary(), udf=concat_udf, udf_name="concat_udf", input_columns=["blob1", "blob2"])

    print("Backfilling UDF...")
    tbl.backfill("concat")
    print("test_backfill_blob done.")


if __name__ == "__main__":
    # 跟这个函数中的类型对应: def _arrow_type_to_sql_type(field: pa.Field) -> str:

    test_backfill_bigint()
    test_backfill_double()
    test_backfill_bool()
    test_backfill_text()
    test_backfill_blob()

