import pyarrow as pa


class _DummyServer:
    def __init__(self):
        self.sqls = []

    def _execute(self, sql: str):
        self.sqls.append(sql)
        return []


class _DummyClientProxy:
    def __init__(self, server):
        self._server = server


class _DummyConn:
    def __init__(self, server):
        self._client_proxy = _DummyClientProxy(server)


def test_add_column_normal():
    from pyseekdb.client.db import Table

    server = _DummyServer()
    conn = _DummyConn(server)
    t = Table(conn, "t1")

    t.add_column("c1", pa.int64(), nullable=False, default=0)

    assert len(server.sqls) == 1
    assert server.sqls[0] == "ALTER TABLE `t1` ADD COLUMN `c1` BIGINT NOT NULL DEFAULT 0"


def test_add_column_generated_virtual():
    from pyseekdb.client.db import Table

    server = _DummyServer()
    conn = _DummyConn(server)
    t = Table(conn, "t1")

    t.add_column("c2", pa.int64(), expression="`c1` + 1", stored=False)

    assert len(server.sqls) == 1
    assert (
        server.sqls[0]
        == "ALTER TABLE `t1` ADD COLUMN `c2` BIGINT GENERATED ALWAYS AS (`c1` + 1) VIRTUAL"
    )


def test_add_column_generated_stored_udf_placeholder():
    from pyseekdb.client.db import Table

    server = _DummyServer()
    conn = _DummyConn(server)
    t = Table(conn, "t1")

    # udf binding not supported yet; should not crash and should still issue DDL
    t.add_column(
        "c3",
        pa.int64(),
        expression="`c1` + 2",
        stored=True,
        udf=lambda x: x,
    )

    assert len(server.sqls) == 1
    assert (
        server.sqls[0]
        == "ALTER TABLE `t1` ADD COLUMN `c3` BIGINT GENERATED ALWAYS AS (`c1` + 2) STORED"
    )


