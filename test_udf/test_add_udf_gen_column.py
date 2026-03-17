import time
import pyarrow as pa

# Ensure we import local pyseekdb from this repo (./src) instead of any installed package.
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

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


def test_add_udf_gen_column_compatibility():
    """
    UDF-generated column only:
    - Skip normal and expression-generated columns
    - Use Python functions (add/mul) via add_column(udf=...)
    - Insert data and validate UDF-generated results
    """
    uri = _load_observer_uri()

    print("Initializing PySeekDB connection (add_udf_gen_column compat)...")
    print("Connecting to", uri)
    try:
        conn = db.connect(uri)
    except Exception as e:
        print("Connection failed, please update uri in test_add_udf_gen_column_compat.py:", e)
        return

    table_name = "test_add_udf_gen_column_" + str(int(time.time()))
    print("Creating table:", table_name)

    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("a", pa.int64()),
            pa.field("b", pa.int64()),
        ]
    )

    data = [{"id": 1, "a": 10, "b": 20}]
    try:
        tbl = conn.create_table(table_name, data=data, schema=schema, mode="overwrite")
    except Exception as e:
        print("Create table failed, please ensure observer is running and uri is correct:", e)
        return

    # Define Python UDFs: sum and product
    def add_udf(a: int, b: int) -> int:
        return a + b

    def mul_udf(a: int, b: int) -> int:
        return a * b

    print("Adding UDF generated column c_sum = a + b (STORED via UDF)...")
    tbl.add_column("c_sum", pa.int64(), udf=add_udf, udf_name="add_udf", input_columns=["a", "b"])

    print("Adding UDF generated column c_prod = a * b (STORED via UDF)...")
    tbl.add_column("c_prod", pa.int64(), udf=mul_udf, udf_name="mul_udf", input_columns=["a", "b"])

    print("Insert another row...")
    tbl.add([{"id": 2, "a": 3, "b": 4}])

    print("Query back and verify UDF generated columns...")
    rows = conn._client_proxy._server._execute(
        f"SELECT `id`,`a`,`b`,`c_sum`,`c_prod` FROM `{table_name}` ORDER BY `id`"
    )
    print("Rows:", rows)

    # Result format depends on driver; normalize to list of tuples
    norm_rows = []
    for r in rows:
        if isinstance(r, dict):
            norm_rows.append(tuple(r[k] for k in ["id", "a", "b", "c_sum", "c_prod"]))
        else:
            norm_rows.append(tuple(r))

    assert len(norm_rows) >= 2
    # id=1 row: sum=30, prod=200
    assert norm_rows[0][0] == 1
    assert norm_rows[0][3] == 30
    assert norm_rows[0][4] == 200
    # id=2 row: sum=7, prod=12
    assert norm_rows[1][0] == 2
    assert norm_rows[1][3] == 7
    assert norm_rows[1][4] == 12

    print("✅ add_udf_gen_column compatibility test passed!")


if __name__ == "__main__":
    test_add_udf_gen_column_compatibility()


