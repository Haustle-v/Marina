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


def test_update_values_with_where():
    from pyseekdb.client.db import Table

    server = _DummyServer()
    conn = _DummyConn(server)
    t = Table(conn, "t1")

    t.update(where="`id` = 1", values={"name": "x", "cnt": 2})

    assert len(server.sqls) == 1
    assert server.sqls[0] == "UPDATE `t1` SET `name` = 'x', `cnt` = 2 WHERE `id` = 1"


def test_update_values_sql():
    from pyseekdb.client.db import Table

    server = _DummyServer()
    conn = _DummyConn(server)
    t = Table(conn, "t1")

    t.update(where="`id` = 1", values_sql={"cnt": "`cnt` + 1"})

    assert len(server.sqls) == 1
    assert server.sqls[0] == "UPDATE `t1` SET `cnt` = `cnt` + 1 WHERE `id` = 1"


def test_update_all_rows_when_where_none():
    from pyseekdb.client.db import Table

    server = _DummyServer()
    conn = _DummyConn(server)
    t = Table(conn, "t1")

    t.update(values_sql={"cnt": "0"})

    assert len(server.sqls) == 1
    assert server.sqls[0] == "UPDATE `t1` SET `cnt` = 0"


def test_update_values_sql_merge_with_values():
    from pyseekdb.client.db import Table

    server = _DummyServer()
    conn = _DummyConn(server)
    t = Table(conn, "t1")

    t.update(where="`id` = 1", values={"name": "y"}, values_sql={"cnt": "`cnt` + 2"})

    assert len(server.sqls) == 1
    assert (
        server.sqls[0]
        == "UPDATE `t1` SET `name` = 'y', `cnt` = `cnt` + 2 WHERE `id` = 1"
    )


