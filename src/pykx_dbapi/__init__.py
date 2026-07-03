"""Arrow-native DB-API wrapper around pykx IPC connections.

Wraps a `pykx.SyncQConnection` so DB-API-aware tools — notably marimo's SQL
cells — can talk to a kdb+/q server. Queries are q/qSQL text sent to the
server verbatim; there is no SQL translation.

The connection also implements the ADBC DB-API extensions
(`fetch_arrow_table`, `adbc_get_objects`, `adbc_get_table_schema`,
`adbc_get_info`), so marimo consumes results as Arrow (near zero-copy to
polars) and can browse tables in its datasources panel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    import pyarrow as pa

apilevel = "2.0"
threadsafety = 1
paramstyle = "qmark"


# PEP 249 exception hierarchy. Errors raised inside pykx (connection
# failures, q-side errors) propagate as pykx exceptions untouched;
# these cover errors raised by this wrapper itself.
class Warning(Exception):  # noqa: A001 - name mandated by PEP 249
    pass


class Error(Exception):
    pass


class InterfaceError(Error):
    pass


class DatabaseError(Error):
    pass


class DataError(DatabaseError):
    pass


class OperationalError(DatabaseError):
    pass


class IntegrityError(DatabaseError):
    pass


class InternalError(DatabaseError):
    pass


class ProgrammingError(DatabaseError):
    pass


class NotSupportedError(DatabaseError):
    pass


def connect(
    *args: Any, arrow_only: bool = True, **kwargs: Any
) -> Connection:
    """Open a connection to a q server.

    Arguments are passed straight through to `pykx.SyncQConnection`,
    e.g. `connect(host="localhost", port=5001)`.

    With `arrow_only=True` (the default), every table result must convert
    to Arrow natively via pykx; results pykx can't convert (keyed tables,
    nested temporal vector columns) raise an error explaining how to
    reshape the query server-side. Pass `arrow_only=False` to instead
    stitch keyed tables from their key/value parts and fall back to a
    Python-object conversion (slower; timestamps at µs precision).

    pykx's context interface is disabled by default: this wrapper never uses
    it, and its k)-dialect bootstrap code breaks some q-only servers (e.g.
    KX Insights). Pass `no_ctx=False` to re-enable.
    """
    import pykx

    kwargs.setdefault("no_ctx", True)
    return Connection(
        pykx.SyncQConnection(*args, **kwargs), arrow_only=arrow_only
    )


def _column(values: list[Any]) -> pa.Array:
    """Arrow array from python values; mixed types fall back to strings."""
    import pyarrow as pa

    try:
        # from_pandas=True treats NaT/NaN as nulls (q null temporals).
        return pa.array(values, from_pandas=True)
    except Exception:
        return pa.array([None if v is None else str(v) for v in values])


def _shape_value(value: Any) -> pa.Table | None:
    """Shape a non-table q result into a one/two-column Arrow table.

    None (q generic null, e.g. an assignment) means "no result set".
    """
    import pyarrow as pa

    if value is None:
        return None
    if isinstance(value, dict):
        return pa.table(
            {
                "key": _column(list(value.keys())),
                "value": _column(list(value.values())),
            }
        )
    if isinstance(value, (list, tuple)):
        return pa.table({"result": _column(list(value))})
    return pa.table({"result": _column([value])})


def _to_arrow(result: Any, arrow_only: bool) -> pa.Table | None:
    """Convert a pykx IPC result to an Arrow table (or None for q null)."""

    try:
        return _convert(result, arrow_only)
    except TypeError as e:
        if "datetime type is deprecated" in str(e):
            # ponytail: clear error over silent wrong data; z has been
            # deprecated since kdb+ 3.0 and pykx won't convert it.
            raise NotSupportedError(
                "the q datetime (z) type is deprecated and unsupported; "
                'cast to timestamp in your query, e.g. "p"$x'
            ) from e
        raise


def _convert(result: Any, arrow_only: bool) -> pa.Table | None:
    import pykx

    if isinstance(result, pykx.KeyedTable):
        if arrow_only:
            raise NotSupportedError(
                "keyed table results cannot be converted to Arrow natively "
                "by pykx; unkey in the query (prefix with 0!, e.g. "
                "`0!select ... by ...`) or use connect(arrow_only=False)"
            )
        # A keyed table is a dictionary of two Tables: convert each and
        # stitch the columns (KeyedTable.pa() is not implemented in pykx).
        import pyarrow as pa

        columns: dict[str, Any] = {}
        for part in (result.keys(), result.values()):
            table = _table_to_arrow(part, arrow_only)
            for name in table.column_names:
                columns[name] = table[name]
        return pa.table(columns)
    if isinstance(result, pykx.Table):
        return _table_to_arrow(result, arrow_only)
    # ponytail: atoms/vectors/dicts round-trip through .py(); fine for the
    # small results these are — tables (the hot path) stay Arrow-native.
    return _shape_value(result.py())


def _table_to_arrow(table: Any, arrow_only: bool) -> pa.Table:
    try:
        return table.pa()
    except TypeError as e:
        # let the deprecated-datetime error bubble to _to_arrow's handler
        if "datetime type is deprecated" in str(e):
            raise
        error: Exception = e
    except Exception as e:
        # pykx's whole-table conversion chokes on some columns (e.g. nested
        # date vectors from `select date, close by ticker`); retry per column.
        error = e
    try:
        return _table_by_column(table)
    except Exception:
        return _pa_failure(table, arrow_only, error)


def _table_by_column(table: Any) -> pa.Table:
    """Per-column Arrow conversion for tables whose whole-table .pa() fails."""
    import pyarrow as pa

    return pa.table(
        {str(name): _column_pa(table[str(name)]) for name in table.columns.py()}
    )


def _column_pa(col: Any) -> pa.Array:
    import pykx
    import pyarrow as pa

    try:
        return col.pa()
    except Exception:
        pass
    if isinstance(col, pykx.List):
        # nested column (one vector per row): convert each sub-vector and
        # stitch into a ListArray.
        parts = [_vector_pa(v) for v in col]
        offsets = [0]
        for part in parts:
            offsets.append(offsets[-1] + len(part))
        return pa.ListArray.from_arrays(
            pa.array(offsets, pa.int32()), pa.concat_arrays(parts)
        )
    return _vector_pa(col)


def _vector_pa(vector: Any) -> pa.Array:
    """Arrow array from a q vector; months/minutes need a numpy-unit cast
    because Arrow has no datetime64[M]/timedelta64[m] equivalent."""
    import pyarrow as pa

    try:
        return vector.pa()
    except Exception:
        array = vector.np()
        if array.dtype.kind == "M":
            return pa.array(array.astype("datetime64[ms]"))
        if array.dtype.kind == "m":
            return pa.array(array.astype("timedelta64[ms]"))
        raise


def _pa_failure(table: Any, arrow_only: bool, error: Exception) -> pa.Table:
    if arrow_only:
        raise NotSupportedError(
            "pykx could not convert this result to Arrow (commonly nested "
            "vector columns of temporals, from a select-by without "
            "aggregation); reshape it server-side (e.g. ungroup) or use "
            "connect(arrow_only=False) to allow a Python-object fallback"
        ) from error
    return _table_from_py(table)


def _table_from_py(table: Any) -> pa.Table:
    """Python-object fallback for tables pykx's .pa() can't convert.

    ponytail: whole-table .py() round-trip; timestamps land at µs (not ns)
    precision here. Per-column .pa() with per-column fallback is the upgrade
    if that ever matters.
    """
    import pyarrow as pa

    return pa.table(
        {str(name): _column(list(v)) for name, v in table.py().items()}
    )


def _arrow_type(type_char: str | bytes) -> pa.DataType:
    """Map a q `meta` type char to an Arrow type."""
    import pyarrow as pa

    char = type_char.decode() if isinstance(type_char, bytes) else type_char
    return {
        "b": pa.bool_(),
        "x": pa.uint8(),
        "h": pa.int16(),
        "i": pa.int32(),
        "j": pa.int64(),
        "e": pa.float32(),
        "f": pa.float64(),
        "c": pa.string(),
        "C": pa.string(),
        "s": pa.string(),
        "g": pa.string(),
        "p": pa.timestamp("ns"),
        "z": pa.timestamp("ms"),
        "d": pa.date32(),
        "m": pa.date32(),
        "n": pa.duration("ns"),
        "u": pa.time32("s"),
        "v": pa.time32("s"),
        "t": pa.time32("ms"),
    }.get(char, pa.null())


class Connection:
    """PEP 249 connection with ADBC extensions, wrapping a pykx connection."""

    # Read by marimo's plain DBAPIEngine, if the ADBC engine is not used.
    dialect = "kdb"

    # kdb has no catalogs/schemas; present one flat namespace.
    adbc_current_catalog = "default"
    adbc_current_db_schema = ""

    def __init__(self, pykx_connection: Any, arrow_only: bool = True) -> None:
        self._q = pykx_connection
        self._arrow_only = arrow_only

    def cursor(self) -> Cursor:
        return Cursor(self._q, self._arrow_only)

    # ponytail: q IPC has no transactions; no-ops satisfy PEP 249.
    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass

    def close(self) -> None:
        self._q.close()

    # --- ADBC extension methods (consumed by marimo's AdbcDBAPIEngine) ---

    def adbc_get_info(self) -> dict[str, Any]:
        return {"vendor_name": "kdb", "driver_name": "pykx-dbapi"}

    def adbc_get_table_schema(
        self, table_name: str, *, db_schema_filter: str | None = None
    ) -> pa.Schema:
        del db_schema_filter
        # meta works on plain, splayed, and partitioned tables alike
        # (taking rows from a bare partitioned table errors with 'par).
        # `meta get t`, not `meta t`: some q servers' meta doesn't
        # dereference symbols.
        meta = self._q("{[t] 0!meta get t}", table_name).py()
        import pyarrow as pa

        # pykx renders the type-char vector as bytes.
        chars = meta["t"]
        if isinstance(chars, bytes):
            chars = chars.decode()
        return pa.schema(
            [
                (str(name), _arrow_type(char))
                for name, char in zip(meta["c"], chars)
            ]
        )

    def adbc_get_objects(
        self,
        *,
        depth: str = "all",
        catalog_filter: str | None = None,
        db_schema_filter: str | None = None,
        table_name_filter: str | None = None,
        table_types_filter: list[str] | None = None,
        column_name_filter: str | None = None,
    ) -> pa.RecordBatchReader:
        del catalog_filter, db_schema_filter, table_types_filter
        del column_name_filter
        import pyarrow as pa

        catalog: dict[str, Any] = {"catalog_name": "default"}
        if depth != "catalogs":
            schema: dict[str, Any] = {"db_schema_name": ""}
            if depth != "db_schemas":
                names = [str(n) for n in self._q("tables[]").py()]
                if table_name_filter is not None:
                    names = [n for n in names if n == table_name_filter]
                schema["db_schema_tables"] = [
                    {"table_name": n, "table_type": "TABLE"} for n in names
                ]
            catalog["catalog_db_schemas"] = [schema]
        return pa.Table.from_pylist([catalog]).to_reader()


class Cursor:
    """PEP 249 cursor. One q round-trip per execute(); results held as Arrow."""

    def __init__(self, q: Any, arrow_only: bool = True) -> None:
        self._q = q
        self._arrow_only = arrow_only
        self.description: list[tuple[Any, ...]] | None = None
        self.rowcount = -1
        self.arraysize = 1
        self._table: pa.Table | None = None
        self._rows: list[tuple[Any, ...]] | None = None
        self._pos = 0

    def execute(
        self, query: str, parameters: Sequence[Any] | None = None
    ) -> Cursor:
        if parameters:
            result = self._q(query, *parameters)
        else:
            result = self._q(query)

        self._table = _to_arrow(result, self._arrow_only)
        self._rows = None
        self._pos = 0
        if self._table is None:
            self.description = None
            self.rowcount = -1
        else:
            self.description = [
                (name, None, None, None, None, None, None)
                for name in self._table.column_names
            ]
            self.rowcount = self._table.num_rows
        return self

    def executemany(
        self, query: str, seq_of_parameters: Sequence[Sequence[Any]]
    ) -> Cursor:
        for parameters in seq_of_parameters:
            self.execute(query, parameters)
        return self

    def fetch_arrow_table(self) -> pa.Table:
        if self._table is None:
            raise InterfaceError("no result set")
        return self._table

    # Tuple-based fetches (plain DB-API consumers, e.g. pandas.read_sql)
    # materialize lazily; marimo's Arrow path never pays for them.

    def _materialized(self) -> list[tuple[Any, ...]]:
        if self._rows is None:
            if self._table is None:
                self._rows = []
            else:
                columns = [c.to_pylist() for c in self._table.columns]
                self._rows = list(zip(*columns)) if columns else []
        return self._rows

    def fetchone(self) -> tuple[Any, ...] | None:
        rows = self._materialized()
        if self._pos >= len(rows):
            return None
        row = rows[self._pos]
        self._pos += 1
        return row

    def fetchmany(self, size: int | None = None) -> list[tuple[Any, ...]]:
        if size is None:
            size = self.arraysize
        rows = self._materialized()[self._pos : self._pos + size]
        self._pos += len(rows)
        return rows

    def fetchall(self) -> list[tuple[Any, ...]]:
        rows = self._materialized()[self._pos :]
        self._pos = len(self._materialized())
        return rows

    # ponytail: PEP 249 requires these; nothing to size or free here.
    def setinputsizes(self, sizes: Sequence[Any]) -> None:
        pass

    def setoutputsize(self, size: int, column: int | None = None) -> None:
        pass

    def close(self) -> None:
        pass
