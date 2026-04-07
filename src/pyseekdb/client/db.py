# SPDX-License-Identifier: Apache-2.0
"""
PySeekDB database client module.
Provides compatibility with lancedb APIs while using PySeekDB's underlying implementation.
"""
import inspect
import logging
import socket
import tempfile
import threading
from typing import Optional, Union, List, Any, Dict, Literal, Iterable, Tuple
import pyarrow as pa
from overrides import override
import numpy as np
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import cloudpickle

from . import Client, AdminClient, HNSWConfiguration
from .collection import Collection
from .embedding_function import EmbeddingFunction, get_default_embedding_function

_LOG = logging.getLogger(__name__)

def _sql_quote_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, np.ndarray)):
        if isinstance(value, np.ndarray):
            value = value.tolist()
        return _sql_quote_string(str(value))
    return _sql_quote_string(str(value))


def _to_arrow_field(name: str, data_type: Any, nullable: bool = True) -> pa.Field:
    if isinstance(data_type, pa.Field):
        return pa.field(data_type.name, data_type.type, nullable=data_type.nullable)
    if isinstance(data_type, pa.DataType):
        return pa.field(name, data_type, nullable=nullable)
    if isinstance(data_type, type):
        if data_type is int:
            return pa.field(name, pa.int64(), nullable=nullable)
        if data_type is float:
            return pa.field(name, pa.float64(), nullable=nullable)
        if data_type is str:
            return pa.field(name, pa.string(), nullable=nullable)
        if data_type is bool:
            return pa.field(name, pa.bool_(), nullable=nullable)
    if isinstance(data_type, str):
        # Best-effort mapping for common SQL type strings.
        t = data_type.strip().upper()
        if "INT" in t:
            return pa.field(name, pa.int64(), nullable=nullable)
        if "DOUBLE" in t or "FLOAT" in t:
            return pa.field(name, pa.float64(), nullable=nullable)
        if "BOOL" in t:
            return pa.field(name, pa.bool_(), nullable=nullable)
        if "BLOB" in t or "BINARY" in t:
            return pa.field(name, pa.binary(), nullable=nullable)
        return pa.field(name, pa.string(), nullable=nullable)
    raise ValueError(f"Unsupported data_type for column '{name}': {data_type!r}")


def _to_sql_type(data_type: Any) -> str:
    if isinstance(data_type, str):
        return data_type.strip()
    field = _to_arrow_field("_tmp", data_type, nullable=True)
    return _arrow_type_to_sql_type(field)


def _ensure_ray_initialized(
    ray_module: Any,
    ray_address: Optional[str],
    ray_init_kwargs: Optional[Dict[str, Any]],
) -> None:
    if ray_module.is_initialized():
        return
    init_kwargs = dict(ray_init_kwargs or {})
    # Default to a light-weight local init if no address is specified.
    if ray_address is not None:
        ray_module.init(address=ray_address, ignore_reinit_error=True, **init_kwargs)
    else:
        # Keep dashboard off by default to avoid extra ports in tests/CI.
        init_kwargs.setdefault("include_dashboard", False)
        ray_module.init(ignore_reinit_error=True, **init_kwargs)

def _arrow_type_to_sql_type(field: pa.Field) -> str:
    """Map PyArrow type to SQL type for CREATE TABLE."""
    """ If any new type is added here, please also add it to the ObTableScanOp::offload_to_ray in ob_table_scan_op.cpp. """
    typ = field.type
    if pa.types.is_string(typ):
        return "TEXT"
    elif pa.types.is_large_string(typ):
        return "LONGTEXT"
    elif pa.types.is_integer(typ):
        return "BIGINT"
    elif pa.types.is_floating(typ):
        return "DOUBLE"
    elif pa.types.is_boolean(typ):
        return "BOOLEAN"
    elif pa.types.is_binary(typ) or pa.types.is_large_binary(typ):
        return "LONGBLOB"   # 512MB in OB
    elif pa.types.is_list(typ) or pa.types.is_large_list(typ) or isinstance(typ, pa.FixedSizeListType):
        # Handle vectors: assume float list is vector if it has fixed size or name implies it
        # For now, simplistic check: if it's a fixed size list of floats, treat as VECTOR
        if isinstance(typ, pa.FixedSizeListType):
            if pa.types.is_float32(typ.value_type) or pa.types.is_float64(typ.value_type):
                return f"VECTOR({typ.list_size})"
        # Fallback for variable list or non-float list -> JSON or TEXT
        return "JSON"
    # Fallback
    return "TEXT"


def _start_temp_http_server(directory: Path) -> Tuple[ThreadingHTTPServer, str]:
    """
    Start a lightweight HTTP server to host files under the provided directory.
    Returns the server instance and base url.
    """
    handler = partial(SimpleHTTPRequestHandler, directory=str(directory))
    httpd = ThreadingHTTPServer(("0.0.0.0", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    try:
        host = socket.gethostbyname(socket.gethostname())
    except Exception:
        host = "127.0.0.1"
    base_url = f"http://{host}:{httpd.server_port}"
    return httpd, base_url


def _write_udf_file_and_serve(udf_func: Any) -> Tuple[str, ThreadingHTTPServer, str]:
    """
    Serialize udf_func via cloudpickle into a temporary file, start HTTP server, return (url, server, file_path).
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="pyseekdb_udf_"))
    file_name = f"{udf_func.__name__}.pkl"
    file_path = temp_dir / file_name
    try:
        data = cloudpickle.dumps(udf_func)
        with open(file_path, "wb") as f:
            f.write(data)
    except Exception:
        pass

    server, base_url = _start_temp_http_server(temp_dir)
    file_url = f"{base_url}/{file_name}"
    return file_url, server, str(file_path)

class Connection:
    """PySeekDB Connection compatible with lancedb.DBConnection."""

    def __init__(
        self,
        uri: str,
        *,
        api_key: Optional[str] = None,
        region: Optional[str] = None,
        host_override: Optional[str] = None,
        **kwargs,
    ) -> None:
        self._uri = uri
        self._api_key = api_key
        self._region = region
        self._host_override = host_override
        self._kwargs = kwargs
        
        # Initialize internal clients
        # Parse URI to determine connection type and parameters
        parsed = urlparse(uri)
        
        client_kwargs = kwargs.copy()
        admin_kwargs = kwargs.copy()
        
        path = None
        host = host_override
        
        # Handle file:// scheme (embedded mode)
        if parsed.scheme == "file" or not parsed.scheme:
            path = parsed.path if parsed.scheme == "file" else uri
            # If uri is just a path string without scheme, treat as path
            if not parsed.scheme and not uri.startswith("/"):
                 # Handle relative path properly if needed, but Client handles it
                 pass
        
        # Handle remote schemes (mysql://, seekdb://)
        elif parsed.scheme in ("mysql", "seekdb"):
            if not host:
                host = parsed.hostname
            
            if parsed.port:
                client_kwargs['port'] = parsed.port
                admin_kwargs['port'] = parsed.port
                
            if parsed.username:
                client_kwargs['user'] = parsed.username
                admin_kwargs['user'] = parsed.username
                
            if parsed.password:
                client_kwargs['password'] = parsed.password
                admin_kwargs['password'] = parsed.password
                
            # Database name usually comes from path, e.g. /dbname
            if parsed.path and parsed.path != "/":
                db_name = parsed.path.lstrip("/")
                client_kwargs['database'] = db_name
            
            # Parse query parameters for additional options like tenant
            if parsed.query:
                params = parse_qs(parsed.query)
                if 'tenant' in params:
                    tenant = params['tenant'][0]
                    client_kwargs['tenant'] = tenant
                    admin_kwargs['tenant'] = tenant
        
        # Initialize clients
        # Client() factory handles embedded vs remote based on path/host presence
        self._client_proxy = Client(path=path, host=host, **client_kwargs)
        self._admin_proxy = AdminClient(path=path, host=host, **admin_kwargs)

    def __repr__(self) -> str:
        return f"<PySeekDB Connection uri={self._uri}>"

    def __enter__(self) -> "Connection":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        """Close the connection."""
        pass

    def table_names(
        self, page_token: Optional[str] = None, limit: Optional[int] = None
    ) -> List[str]:
        """List all available tables."""
        # Using list_collections for now, but if we create custom tables, we might want to query information_schema directly
        # The underlying _execute method is available on the server instance
        try:
            # Try to use raw SQL to list tables if possible
            tables = self._client_proxy._server._execute("SHOW TABLES")
            # Format depends on client type (embedded returns tuples, server returns dicts or tuples)
            names = []
            for row in tables:
                if isinstance(row, dict):
                    names.append(list(row.values())[0])
                elif isinstance(row, (tuple, list)):
                    names.append(row[0])
                else:
                    names.append(str(row))
            return names
        except Exception as e:
            _LOG.warning(f"Failed to list tables via SQL: {e}. Falling back to list_collections")
            collections = self._client_proxy.list_collections()
            return [c.name for c in collections]

    def open_table(self, name: str) -> "Table":
        """Open a table."""
        # We return a Table wrapper. Since we support custom schemas now, 
        # we don't necessarily call get_collection.
        return Table(self, name)

    def create_table(
        self,
        name: str,
        data: Optional[Any] = None,
        schema: Optional[Any] = None,
        mode: str = "create",
        exist_ok: bool = False,
        on_bad_vectors: str = "error",
        fill_value: float = 0.0,
        **kwargs,
    ) -> "Table":
        """Create a relational table with custom schema."""
        # Determine schema
        arrow_schema = None
        if schema is not None:
            if isinstance(schema, pa.Schema):
                arrow_schema = schema
            # TODO: Support LanceModel pydantic to schema conversion if needed
        elif data is not None:
            # Infer from data
            try:
                if isinstance(data, list):
                    arrow_schema = pa.Table.from_pylist(data).schema
                elif hasattr(data, 'to_arrow'): # pandas df often supports this via pyarrow
                    arrow_schema = pa.Table.from_pandas(data).schema
                elif isinstance(data, pa.Table):
                    arrow_schema = data.schema
            except Exception as e:
                raise ValueError(f"Failed to infer schema from data: {e}")
        
        if arrow_schema is None:
            raise ValueError("Must provide either 'data' or 'schema' to create a table")

        # Handle existing table
        try:
            # Check if table exists
            self._client_proxy._server._execute(f"DESCRIBE `{name}`")
            exists = True
        except Exception:
            exists = False

        if mode == "overwrite" and exists:
            self.drop_table(name)
            exists = False
        elif mode == "create" and exists:
            if exist_ok:
                return self.open_table(name)
            raise ValueError(f"Table '{name}' already exists")

        # Generate CREATE TABLE SQL
        columns_sql = []
        vector_cols = []
        for field in arrow_schema:
            sql_type = _arrow_type_to_sql_type(field)
            col_def = f"`{field.name}` {sql_type}"
            columns_sql.append(col_def)
            if "VECTOR" in sql_type:
                vector_cols.append(field.name)
        
        sql = f"CREATE TABLE `{name}` ({', '.join(columns_sql)})"
        
        _LOG.info(f"Creating table '{name}' with SQL: {sql}")
        self._client_proxy._server._execute(sql)
        
        table = Table(self, name, schema=arrow_schema)
        
        if data is not None:
            table.add(data, on_bad_vectors=on_bad_vectors, fill_value=fill_value)
            
        return table

    def create_view(self, name: str, query: str, materialized: bool = False) -> "Table":
        """Create a View."""
        if materialized:
            raise NotImplementedError("Materialized views are not fully supported yet.")
        sql = f"CREATE VIEW `{name}` AS {query}"
        self._client_proxy._server._execute(sql)
        return Table(self, name)

    def create_materialized_view(
        self,
        name: str,
        query: Any, 
        with_no_data: bool = True,
    ) -> "Table":
        raise NotImplementedError("Materialized views are not currently supported in PySeekDB.")

    def drop_view(self, name: str) -> None:
        self._client_proxy._server._execute(f"DROP VIEW IF EXISTS `{name}`")

    def drop_table(self, name: str, *args, **kwargs) -> None:
        """Drop a table."""
        self._client_proxy._server._execute(f"DROP TABLE IF EXISTS `{name}`")

class Table:
    """Table wrapper for generic Relational Table in PySeekDB."""

    def __init__(self, conn: Connection, name: str, schema: Optional[pa.Schema] = None) -> None:
        self._conn = conn
        self._name = name
        self._schema = schema

    @property
    def name(self) -> str:
        return self._name

    @property
    def schema(self) -> pa.Schema:
        """Return the schema of the table."""
        if self._schema:
            return self._schema
        # Try to infer from DB if not cached
        return pa.schema([]) 

    def add(
        self,
        data: Any,
        mode: str = "append",
        on_bad_vectors: str = "error",
        fill_value: float = 0.0,
    ) -> None:
        """Add data to the table using INSERT."""
        import pandas as pd
        
        # Convert data to list of dicts
        rows = []
        if isinstance(data, list):
            rows = data
        elif isinstance(data, pd.DataFrame):
            rows = data.to_dict(orient='records')
        elif isinstance(data, pa.Table):
            rows = data.to_pylist()
            
        if not rows:
            return

        # Generate INSERT SQL
        keys = list(rows[0].keys())
        columns_str = ", ".join([f"`{k}`" for k in keys])
        
        values_list = []
        for row in rows:
            vals = []
            for k in keys:
                val = row.get(k)
                if val is None:
                    vals.append("NULL")
                elif isinstance(val, (int, float)):
                    vals.append(str(val))
                elif isinstance(val, bool):
                    vals.append("1" if val else "0")
                elif isinstance(val, bytes):
                    vals.append(f"X'{val.hex()}'")
                elif isinstance(val, (list, np.ndarray)):
                    # Vector or list -> string representation
                    if isinstance(val, np.ndarray):
                        val = val.tolist()
                    vals.append(_sql_quote_string(str(val)))
                else:
                    vals.append(_sql_quote_string(str(val)))
            values_list.append(f"({', '.join(vals)})")
            
        # Bulk insert
        batch_size = 100
        for i in range(0, len(values_list), batch_size):
            batch = values_list[i:i+batch_size]
            sql = f"INSERT INTO `{self._name}` ({columns_str}) VALUES {', '.join(batch)}"
            self._conn._client_proxy._server._execute(sql)
            
        _LOG.info(f"Added {len(rows)} rows to table '{self._name}'")

    def search(
        self,
        query: Optional[Any] = None,
        vector_column_name: Optional[str] = None,
        query_type: Literal["vector", "fts", "hybrid", "auto"] = "auto",
        ordering_field_name: Optional[str] = None,
        fts_columns: Optional[Union[str, List[str]]] = None,
    ) -> Any:
        """Perform search using SQL."""
        return GenevaQueryBuilder(self, query, query_type, vector_column_name)

    def create_vector_index(
        self,
        metric: str = "L2",
        vector_column_name: str = "vector",
        replace: bool = True,
        index_type: str = "IVF_PQ", 
        **kwargs,
    ) -> None:
        """Create Vector Index."""
        distance = metric.lower() if metric else "l2"
        index_name = f"idx_vec_{vector_column_name}"
        sql = f"CREATE VECTOR INDEX `{index_name}` ON `{self._name}` (`{vector_column_name}`) WITH (DISTANCE={distance}, TYPE=HNSW, LIB=VSAG)"
        _LOG.info(f"Creating vector index: {sql}")
        try:
            self._conn._client_proxy._server._execute(sql)
        except Exception as e:
            _LOG.error(f"Failed to create index: {e}")
            raise

    def create_fts_index(
        self,
        field_names: Union[str, List[str]],
        replace: bool = False,
        **kwargs,
    ) -> None:
        """Create Full Text Search Index."""
        if isinstance(field_names, str):
            field_names = [field_names]
        
        for field in field_names:
            index_name = f"idx_fts_{field}"
            sql = f"CREATE FULLTEXT INDEX `{index_name}` ON `{self._name}`(`{field}`)"
            _LOG.info(f"Creating FTS index: {sql}")
            self._conn._client_proxy._server._execute(sql)

    def create_scalar_index(
        self,
        column: str,
        replace: bool = True,
        index_type: str = "BTREE",
    ) -> None:
        """Create Scalar Index."""
        index_name = f"idx_{column}"
        sql = f"CREATE INDEX `{index_name}` ON `{self._name}`(`{column}`)"
        _LOG.info(f"Creating scalar index: {sql}")
        self._conn._client_proxy._server._execute(sql)

    def drop_columns(self, columns: Iterable[str]) -> None:
        """Drop columns."""
        for col in columns:
            sql = f"ALTER TABLE `{self._name}` DROP COLUMN `{col}`"
            self._conn._client_proxy._server._execute(sql)

    def add_column(
        self,
        col_name: str,
        data_type: Any,
        *,
        expression: Optional[str] = None,
        stored: bool = True,
        nullable: bool = True,
        default: Any = None,
        comment: Optional[str] = None,
        udf: Any = None,
        udf_name: Optional[str] = None, 
        input_columns: Optional[List[str]] = None,
    ) -> None:
        """
        Add a new column to this table.

        This is a table-level API similar to geneva-0.7.0's add_columns for UDF columns.

        Supported:
        - Normal columns:           `ALTER TABLE tbl ADD COLUMN col_name sql_type NOT NULL`
        - Normal Generated columns: `ALTER TABLE tbl ADD COLUMN col_name sql_type GENERATED ALWAYS AS (<expression>) [VIRTUAL|STORED]`
        - UDF Generated columns:    `ALTER TABLE tbl ADD COLUMN col_name sql_type GENERATED ALWAYS AS (<udf_name>) [VIRTUAL|STORED]`
        """
        if not col_name or not isinstance(col_name, str):
            raise ValueError("col_name must be a non-empty string")

        col_def = f"`{col_name}` {_to_sql_type(data_type)}"
        server: Optional[ThreadingHTTPServer] = None
        udf_file_path: Optional[Path] = None
        udf_temp_dir: Optional[Path] = None

        if expression is not None:
            # Normal expression generated columns
            expr = expression.strip()
            if not expr:
                raise ValueError("expression must be a non-empty string when provided")
            col_def += f" GENERATED ALWAYS AS ({expr})"
            col_def += " STORED" if stored else " VIRTUAL"
        elif udf is not None:
            # UDF generated columns
            if not inspect.isfunction(udf):
                raise ValueError("udf must be a Python function")

            # file_url: the url of the udf file uploaded to the server
            # file_path: the path of the temporary udf file
            file_url, server, file_path = _write_udf_file_and_serve(udf)
            udf_file_path = Path(file_path)
            udf_temp_dir = udf_file_path.parent
             # udf_name is the routine_name of the udf routine. It would be set to the name of udf if not explicitly provided.
            if udf_name is None:
                udf_name = udf.__name__

            joined_input_columns = ""
            if input_columns:
                joined_input_columns = ",".join(input_columns)

            # udf_name is the only identifier for the udf function
            create_udf_func_sql = (
                f"CREATE FUNCTION {udf_name}(arg1 INT) "
                "RETURNS INT "
                "PROPERTIES ("
                f"symbol = {_sql_quote_string(udf_name)}, "
                "type = 'Python', "
                f"file = {_sql_quote_string(file_url)}, "
                "mode = 'remote',"
                f"input_columns = {_sql_quote_string(joined_input_columns)}"
                ");"
            )
            _LOG.info("Creating UDF with SQL: %s; source file: %s", create_udf_func_sql, file_path)

            # Best-effort cleanup for existing same-name UDF before CREATE.
            drop_udf_if_exist_sql = f"DROP FUNCTION IF EXISTS {udf_name};"
            self._conn._client_proxy._server._execute(drop_udf_if_exist_sql)

            # execute the sql to create the udf function
            self._conn._client_proxy._server._execute(create_udf_func_sql)

            col_def += f" AS UDF {udf_name}"

        else:
            # Normal columns
            col_def += " NULL" if nullable else " NOT NULL"
            if default is not None:
                col_def += f" DEFAULT {_sql_literal(default)}"

        if comment is not None:
            col_def += f" COMMENT {_sql_quote_string(comment)}"

        sql = f"ALTER TABLE `{self._name}` ADD COLUMN {col_def}"
        _LOG.info("Adding column with SQL: %s", sql)
        self._conn._client_proxy._server._execute(sql)

        # Cleanup temp UDF server and files
        if server is not None:
            try:
                server.shutdown()
                server.server_close()
            except Exception:
                pass
        if udf_file_path is not None:
            try:
                udf_file_path.unlink(missing_ok=True)
            except Exception:
                pass
        if udf_temp_dir is not None:
            try:
                if not any(udf_temp_dir.iterdir()):
                    udf_temp_dir.rmdir()
            except Exception:
                pass

        # Best-effort schema cache update
        try:
            if self._schema is not None:
                arrow_field = _to_arrow_field(col_name, data_type, nullable=nullable)
                self._schema = self._schema.append(arrow_field)
        except Exception:
            # Schema cache is optional; ignore failures
            pass

    def backfill(self, col_name: str, num_gpus: int = 0, num_batches: int = 1) -> None:
        ray_offload_hint = "/*+ USE_RAY_OFFLOAD"
        if num_gpus != 0:
            ray_offload_hint += f" USE_RAY_OFFLOAD_NUM_GPUS({num_gpus})"
        if num_batches != 0:
            ray_offload_hint += f" USE_RAY_OFFLOAD_NUM_BATCHES({num_batches})"
        ray_offload_hint += f" QUERY_TIMEOUT(1000000000)*/"

        backfill_sql = (
            f"UPDATE {ray_offload_hint} {self._name} SET {col_name} = NULL;"
        )
        _LOG.info("Backfill with SQL: %s", backfill_sql)
        self._conn._client_proxy._server._execute(backfill_sql)

    def delete(self, where: str) -> None:
        """Delete rows based on filter."""
        sql = f"DELETE FROM `{self._name}` WHERE {where}"
        self._conn._client_proxy._server._execute(sql)

    def update(
        self,
        where: Optional[str] = None,
        values: Optional[Dict[str, Any]] = None,
        *,
        values_sql: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Update rows in the table.

        Keep API consistent with geneva-0.7.0:
          Table.update(where: str | None = None,
                       values: dict | None = None,
                       *,
                       values_sql: dict[str, str] | None = None) -> None

        Notes:
        - values: Python values will be converted to SQL literals.
        - values_sql: SQL expressions inserted as-is (no quoting), e.g. {"cnt": "cnt + 1"}.
        - If where is None, all rows will be updated.
        """
        if values is None and values_sql is None:
            raise ValueError("Either values or values_sql must be provided")
        if values is not None and not isinstance(values, dict):
            raise ValueError("values must be a dict when provided")
        if values_sql is not None and not isinstance(values_sql, dict):
            raise ValueError("values_sql must be a dict when provided")

        sets: List[str] = []
        if values:
            for k, v in values.items():
                sets.append(f"`{k}` = {_sql_literal(v)}")
        if values_sql:
            for k, v in values_sql.items():
                if v is None:
                    raise ValueError(f"values_sql for column '{k}' must be a string")
                sets.append(f"`{k}` = {str(v)}")

        if not sets:
            raise ValueError("No columns to update")

        sql = f"UPDATE `{self._name}` SET " + ", ".join(sets)
        if where is not None:
            w = str(where).strip()
            if not w:
                raise ValueError("where must be a non-empty string when provided")
            sql += f" WHERE {w}"

        _LOG.info("Updating rows with SQL: %s", sql)
        self._conn._client_proxy._server._execute(sql)


def connect(uri: str, **kwargs) -> Connection:
    return Connection(uri, **kwargs)


class GenevaQueryBuilder:
    """SQL QueryBuilder for PySeekDB."""
    def __init__(self, table: Table, query: Any, query_type: str, vector_col: str = None):
        self._table = table
        self._query = query
        self._query_type = query_type
        self._vector_col = vector_col
        self._limit = 10
        self._where = None

    def limit(self, limit: int) -> "GenevaQueryBuilder":
        self._limit = limit
        return self

    def where(self, where: str) -> "GenevaQueryBuilder":
        self._where = where
        return self

    def to_pandas(self):
        import pandas as pd
        
        # Construct SQL
        sql = f"SELECT * FROM `{self._table.name}`"
        
        if self._where:
            sql += f" WHERE {self._where}"
            
        if self._query is not None and self._vector_col:
            vec_str = str(self._query)
            # OceanBase syntax for vector distance
            sql += f" ORDER BY l2_distance(`{self._vector_col}`, '{vec_str}')"
        
        sql += f" LIMIT {self._limit}"
        
        try:
            rows = self._table._conn._client_proxy._server._execute(sql)
            if not rows:
                return pd.DataFrame()
            
            # Use schema names if available and rows are tuples
            if isinstance(rows[0], (tuple, list)):
                columns = None
                # Try to get column names from table schema if available
                if hasattr(self._table, 'schema') and self._table.schema:
                     if len(self._table.schema.names) == len(rows[0]):
                         columns = self._table.schema.names
                return pd.DataFrame(rows, columns=columns)
            
            return pd.DataFrame(rows)
            
        except Exception as e:
            _LOG.error(f"Query failed: {e}")
            raise

    def to_arrow(self) -> pa.Table:
        df = self.to_pandas()
        return pa.Table.from_pandas(df)
