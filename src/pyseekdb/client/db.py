# SPDX-License-Identifier: Apache-2.0
"""
PySeekDB database client module.
Provides compatibility with lancedb APIs while using PySeekDB's underlying implementation.
"""
import logging
from typing import Optional, Union, List, Any, Dict, Literal, Iterable
import pyarrow as pa
from overrides import override
import numpy as np
from urllib.parse import urlparse, parse_qs
import json
import uuid
import time

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
        return "BLOB"
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
                elif isinstance(val, (list, np.ndarray)):
                    # Vector or list -> string representation
                    if isinstance(val, np.ndarray):
                        val = val.tolist()
                    vals.append(f"'{str(val)}'")
                else:
                    safe_val = str(val).replace("'", "''")
                    vals.append(f"'{safe_val}'")
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
        name: str,
        data_type: Any,
        *,
        expression: Optional[str] = None,
        stored: bool = False,
        nullable: bool = True,
        default: Any = None,
        comment: Optional[str] = None,
        udf: Any = None,
        input_columns: Optional[List[str]] = None,
    ) -> None:
        """
        Add a new column to this table.

        This is a table-level API similar to geneva-0.7.0's add_columns for UDF columns.

        Supported:
        - Normal columns: `ALTER TABLE ... ADD COLUMN ...`
        - Generated columns: `GENERATED ALWAYS AS (<expression>) [VIRTUAL|STORED]`

        Not supported (yet):
        - Associating generated columns with a Python UDF (OceanBase generated columns
          do not support UDF today). The `udf` parameter is reserved for future use.
        """
        if not name or not isinstance(name, str):
            raise ValueError("name must be a non-empty string")

        sql_type = _to_sql_type(data_type)
        col_def = f"`{name}` {sql_type}"

        if expression is not None:
            expr = expression.strip()
            if not expr:
                raise ValueError("expression must be a non-empty string when provided")
            col_def += f" GENERATED ALWAYS AS ({expr})"
            col_def += " STORED" if stored else " VIRTUAL"

            # TODO: OceanBase generated columns do not support UDF today.
            # Keep this parameter for future extension (e.g. encode udf metadata
            # or translate to server-side UDF when supported).
            if udf is not None:
                _LOG.warning(
                    "add_column udf binding is not supported yet; ignoring udf=%r",
                    udf,
                )
        else:
            col_def += " NULL" if nullable else " NOT NULL"
            if default is not None:
                col_def += f" DEFAULT {_sql_literal(default)}"

        if comment is not None:
            col_def += f" COMMENT {_sql_quote_string(comment)}"

        sql = f"ALTER TABLE `{self._name}` ADD COLUMN {col_def}"
        _LOG.info("Adding column with SQL: %s", sql)
        self._conn._client_proxy._server._execute(sql)

        # Best-effort schema cache update
        try:
            if self._schema is not None:
                arrow_field = _to_arrow_field(name, data_type, nullable=nullable)
                self._schema = self._schema.append(arrow_field)
        except Exception:
            # Schema cache is optional; ignore failures
            pass

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

    def backfill_async(
        self,
        col_name: str,
        *,
        udf: Any = None,
        where: Optional[str] = None,
        ray_address: Optional[str] = None,
        ray_init_kwargs: Optional[Dict[str, Any]] = None,
        _enable_job_tracker_saves: bool = True,
        **kwargs,
    ) -> Any:
        """
        Backfill a column asynchronously (Ray-based), similar to geneva-0.7.0.

        Notes:
        - Unlike Geneva/Lance, pyseekdb backfill writes results back to an OceanBase
          relational table via UPDATE statements.
        - Column must already exist. Use Table.add_column first.
        - Currently the UDF must be provided explicitly (pyseekdb does not persist
          UDF specs in table metadata like Geneva does).

        Extra kwargs (best-effort supported):
        - batch_size: int (default 100)
        - concurrency: int (default 8)
        - key_column: str (default "id") used to identify rows for UPDATE
        - read_columns: list[str] | None (override input columns)
        """
        if not col_name or not isinstance(col_name, str):
            raise ValueError("col_name must be a non-empty string")
        if udf is None:
            raise ValueError("udf must be provided for pyseekdb backfill")

        try:
            import ray  # type: ignore
        except Exception as e:
            raise ImportError(
                "ray is required for backfill_async. Please install it via `pip install ray`."
            ) from e

        batch_size = int(kwargs.get("batch_size", 100) or 100)
        concurrency = int(kwargs.get("concurrency", 8) or 8)
        key_column = str(kwargs.get("key_column", "id"))
        read_columns = kwargs.get("read_columns", None)

        # Validate column exists (server-side best-effort)
        try:
            self._conn._client_proxy._server._execute(f"DESCRIBE `{self._name}`")
        except Exception as e:
            raise RuntimeError(f"Failed to describe table `{self._name}`: {e}") from e

        # Determine input columns from UDF
        try:
            from pyseekdb.transformer import UDF as SeekUDF  # optional import
        except Exception:
            SeekUDF = None  # type: ignore

        udf_obj = udf
        if SeekUDF is not None and isinstance(udf_obj, SeekUDF):
            input_cols = udf_obj.input_columns
        else:
            # Assume callable; try to access .input_columns if present
            input_cols = getattr(udf_obj, "input_columns", None)

        if read_columns is not None:
            input_cols = list(read_columns)

        if input_cols is None:
            # RecordBatch UDF in Geneva uses full batch; here we must still select columns.
            raise ValueError(
                "Unable to infer input columns for UDF. Please pass read_columns=[...]"
            )

        job_id = uuid.uuid4().hex

        # Initialize/attach Ray if not already running.
        # If ray_address is provided, attach to that cluster.
        _ensure_ray_initialized(ray, ray_address, ray_init_kwargs)

        @ray.remote  # type: ignore[misc]
        def _backfill_batch_task(
            uri: str,
            table_name: str,
            target_col: str,
            key_col: str,
            input_cols_task: List[str],
            udf_bytes: bytes,
            where_sql: Optional[str],
            offset: int,
            limit: int,
        ) -> int:
            import cloudpickle
            import pymysql
            import pyarrow as pa

            parsed = urlparse(uri)
            host = parsed.hostname
            port = parsed.port or 3306
            user = parsed.username or "root"
            password = parsed.password or ""
            database = parsed.path.lstrip("/") if parsed.path and parsed.path != "/" else None
            params = parse_qs(parsed.query or "")
            tenant = params.get("tenant", [None])[0]

            udf_local = cloudpickle.loads(udf_bytes)

            # OceanBase/SeekDB tenant is typically encoded as user@tenant for authentication.
            # Using session variables (e.g. SET @ob_tenant) is not reliable across server setups.
            full_user = user
            if tenant and user and "@" not in user:
                full_user = f"{user}@{tenant}"

            conn = pymysql.connect(
                host=host,
                port=port,
                user=full_user,
                password=password,
                database=database,
                autocommit=True,
            )
            try:
                with conn.cursor() as cur:
                    # Phase 1: scan ONLY primary key (or key column) for batching.
                    # This minimizes data transfer and avoids pulling large columns
                    # (e.g. blobs) just for pagination.
                    key_sql = f"SELECT `{key_col}` FROM `{table_name}`"
                    if where_sql:
                        key_sql += f" WHERE {where_sql}"
                    key_sql += (
                        f" ORDER BY `{key_col}` LIMIT {int(limit)} OFFSET {int(offset)}"
                    )
                    cur.execute(key_sql)
                    key_rows = cur.fetchall()
                    if not key_rows:
                        return 0

                    keys = [r[0] for r in key_rows]

                    # Phase 2: fetch only the required input columns for these keys.
                    # Note: we do not rely on ORDER BY here; we will map by key later.
                    cols = [key_col] + list(input_cols_task)
                    select_cols_sql = ", ".join([f"`{c}`" for c in cols])
                    in_list = ", ".join([_sql_literal(k) for k in keys])
                    sql = (
                        f"SELECT {select_cols_sql} FROM `{table_name}` "
                        f"WHERE `{key_col}` IN ({in_list})"
                    )
                    cur.execute(sql)
                    rows = cur.fetchall()
                    if not rows:
                        return 0

                    # Normalize rows to dicts
                    desc = [d[0] for d in cur.description]
                    dict_rows = [dict(zip(desc, r)) for r in rows]

                    # Build record batch
                    batch = pa.RecordBatch.from_pylist(dict_rows)

                    # Apply UDF (supports our migrated UDF wrapper and plain callables)
                    try:
                        out_arr = udf_local(batch)  # record-batch UDF style
                    except TypeError:
                        # Fallback: scalar UDF expecting python values per row
                        out_vals = []
                        for r in dict_rows:
                            args = [r[c] for c in input_cols_task]
                            out_vals.append(udf_local(*args))
                        out_arr = pa.array(out_vals)

                    out_vals = out_arr.to_pylist()
                    keys = [r[key_col] for r in dict_rows]

                    # Build one UPDATE with CASE WHEN to reduce round-trips
                    case_parts = []
                    in_parts = []
                    for k, v in zip(keys, out_vals):
                        in_parts.append(str(int(k)) if isinstance(k, (int,)) else _sql_literal(k))
                        if v is None:
                            v_sql = "NULL"
                        elif isinstance(v, (dict, list)):
                            v_sql = _sql_literal(json.dumps(v))
                        else:
                            v_sql = _sql_literal(v)
                        case_parts.append(f"WHEN { _sql_literal(k) } THEN {v_sql}")

                    update_sql = (
                        f"UPDATE `{table_name}` SET `{target_col}` = CASE `{key_col}` "
                        + " ".join(case_parts)
                        + " END WHERE `"
                        + key_col
                        + "` IN ("
                        + ", ".join([_sql_literal(k) for k in keys])
                        + ")"
                    )
                    cur.execute(update_sql)
                    return len(keys)
            finally:
                conn.close()

        # Serialize UDF for workers
        try:
            import cloudpickle
        except Exception as e:
            raise ImportError("cloudpickle is required for backfill_async") from e
        udf_payload = cloudpickle.dumps(udf_obj)

        # Total row count
        count_sql = f"SELECT COUNT(*) AS cnt FROM `{self._name}`"
        if where:
            count_sql += f" WHERE {where}"
        res = self._conn._client_proxy._server._execute(count_sql)
        if isinstance(res, list) and res:
            if isinstance(res[0], dict):
                total = int(list(res[0].values())[0])
            else:
                total = int(res[0][0])
        else:
            total = 0

        offsets = list(range(0, total, batch_size))
        # Limit concurrency by submitting all tasks; Ray will schedule accordingly.
        obj_refs = [
            _backfill_batch_task.remote(
                self._conn._uri,
                self._name,
                col_name,
                key_column,
                list(input_cols),
                udf_payload,
                where,
                off,
                batch_size,
            )
            for off in offsets
        ]

        class RayJobFuture:
            def __init__(self, job_id: str, refs: List[Any]) -> None:
                self.job_id = job_id
                self._refs = refs

            def done(self, timeout: Optional[float] = None) -> bool:
                if not self._refs:
                    return True
                ready, _ = ray.wait(self._refs, num_returns=len(self._refs), timeout=timeout)
                return len(ready) == len(self._refs)

            def result(self, timeout: Optional[float] = None) -> Any:
                # timeout best-effort: if provided, wait then raise
                if timeout is not None and not self.done(timeout=timeout):
                    raise TimeoutError("Backfill job not completed within timeout")
                return ray.get(self._refs)

            def status(self, timeout: Optional[float] = None) -> None:
                if not self._refs:
                    print(f"job {self.job_id}: no work")
                    return
                ready, _ = ray.wait(self._refs, num_returns=len(self._refs), timeout=0.0)
                print(f"job {self.job_id}: {len(ready)}/{len(self._refs)} batches done")

        return RayJobFuture(job_id, obj_refs)

    def backfill(
        self,
        col_name: str,
        *,
        udf: Any = None,
        where: Optional[str] = None,
        concurrency: int = 8,
        intra_applier_concurrency: int = 1,
        refresh_status_secs: float = 2.0,
        ray_address: Optional[str] = None,
        ray_init_kwargs: Optional[Dict[str, Any]] = None,
        _enable_job_tracker_saves: bool = True,
        **kwargs,
    ) -> str:
        """
        Backfill a column synchronously (blocking), similar to geneva-0.7.0.

        Returns job_id string.
        """
        # Keep signature compatible; intra_applier_concurrency not used in SQL backend yet.
        fut = self.backfill_async(
            col_name,
            udf=udf,
            where=where,
            concurrency=concurrency,
            intra_applier_concurrency=intra_applier_concurrency,
            ray_address=ray_address,
            ray_init_kwargs=ray_init_kwargs,
            _enable_job_tracker_saves=_enable_job_tracker_saves,
            **kwargs,
        )
        while not fut.done(timeout=refresh_status_secs):
            fut.status()
            time.sleep(max(0.1, float(refresh_status_secs)))
        fut.status()
        fut.result()
        return fut.job_id

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
