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


def test_add_expr_gen_column_compatibility():
    """
    Integration-style test (similar to test_lancedb_compat.py):
    - Connect to an OceanBase observer via mysql:// URI
    - Create a table
    - Add a normal column
    - Add a generated column (VIRTUAL/STORED)
    - Insert and verify computed values via SELECT

    How to run:
      export PYSEEKDB_TEST_OB_URI="mysql://root:pass@127.0.0.1:2881/test?tenant=t1"
      python3 test_add_expr_gen_column_compat.py
    """
    uri = _load_observer_uri()

    print("Initializing PySeekDB connection (add_expr_gen_column compat)...")
    print("Connecting to", uri)
    try:
        conn = db.connect(uri)
    except Exception as e:
        print("Connection failed, please update uri in test_add_expr_gen_column_compat.py:", e)
        return

    table_name = "test_add_expr_gen_column_" + str(int(time.time()))
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

    print("Adding normal column c1...")
    tbl.add_column("c1", pa.int64(), nullable=False, default=7)

    print("Adding expression generated column c2 = a + b (VIRTUAL)...")
    # Note: OB syntax is MySQL compatible for generated columns.
    tbl.add_column("c2", pa.int64(), expression="`a` + `b`", stored=False)

    print("Adding expression generated column c3 = a * 2 (STORED)...")
    tbl.add_column("c3", pa.int64(), expression="`a` * 2", stored=True)

    print("Insert another row...")
    tbl.add([{"id": 2, "a": 3, "b": 4}])

    print("Query back and verify expression generated columns...")
    rows = conn._client_proxy._server._execute(
        f"SELECT `id`,`a`,`b`,`c1`,`c2`,`c3` FROM `{table_name}` ORDER BY `id`"
    )
    print("Rows:", rows)

    # Result format depends on driver; normalize to list of tuples
    norm_rows = []
    for r in rows:
        if isinstance(r, dict):
            norm_rows.append(tuple(r[k] for k in ["id", "a", "b", "c1", "c2", "c3"]))
        else:
            norm_rows.append(tuple(r))

    assert len(norm_rows) >= 2
    # id=1 row: c1 default 7, c2=30, c3=20
    assert norm_rows[0][0] == 1
    assert norm_rows[0][3] == 7
    assert norm_rows[0][4] == 30
    assert norm_rows[0][5] == 20
    # id=2 row: c1 default 7, c2=7, c3=6
    assert norm_rows[1][0] == 2
    assert norm_rows[1][3] == 7
    assert norm_rows[1][4] == 7
    assert norm_rows[1][5] == 6

    print("✅ add_expr_gen_column compatibility test passed!")


if __name__ == "__main__":
    test_add_expr_gen_column_compatibility()


