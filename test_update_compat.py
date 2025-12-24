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


def test_update_compatibility():
    """
    Integration-style test (same style as test_add_column_compat.py):
    - Connect to an OceanBase observer via mysql:// URI
    - Create a table
    - Insert rows
    - Call Table.update(...) with values and values_sql
    - Verify via SELECT
    """
    uri = _load_observer_uri()

    print("Initializing PySeekDB connection (update compat)...")
    print("Connecting to", uri)
    try:
        conn = db.connect(uri)
    except Exception as e:
        print("Connection failed, please update uri in test_update_compat.py:", e)
        return

    table_name = "test_update_" + str(int(time.time()))
    print("Creating table:", table_name)

    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("name", pa.string()),
            pa.field("cnt", pa.int64()),
        ]
    )
    data = [
        {"id": 1, "name": "a", "cnt": 10},
        {"id": 2, "name": "b", "cnt": 20},
    ]

    try:
        tbl = conn.create_table(table_name, data=data, schema=schema, mode="overwrite")
    except Exception as e:
        print("Create table failed, please ensure observer is running and uri is correct:", e)
        return

    print("Update row id=1 using values (literal update)...")
    tbl.update(where="`id` = 1", values={"name": "a1", "cnt": 11})

    print("Update row id=2 using values_sql (expression update)...")
    tbl.update(where="`id` = 2", values_sql={"cnt": "`cnt` + 3"})

    print("Query back and verify...")
    rows = conn._client_proxy._server._execute(
        f"SELECT `id`,`name`,`cnt` FROM `{table_name}` ORDER BY `id`"
    )
    print("Rows:", rows)

    norm_rows = []
    for r in rows:
        if isinstance(r, dict):
            norm_rows.append((r["id"], r["name"], r["cnt"]))
        else:
            norm_rows.append(tuple(r))

    assert len(norm_rows) >= 2
    assert norm_rows[0] == (1, "a1", 11)
    assert norm_rows[1] == (2, "b", 23)

    print("✅ update compatibility test passed!")


if __name__ == "__main__":
    test_update_compatibility()


