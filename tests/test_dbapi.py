"""Tests against a fake pykx module — no q server or pykx install needed."""

from __future__ import annotations

import sys
import types
from typing import Any

import pandas as pd
import pyarrow as pa
import pytest

import pykx_dbapi


class FakeTable:
    """pykx.Table stand-in; pandas only as a test-side convenience."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def pa(self) -> pa.Table:
        return pa.Table.from_pandas(self._df, preserve_index=False)

    def py(self) -> dict[str, list[Any]]:
        return {
            str(c): [
                v.tolist() if hasattr(v, "tolist") else v for v in self._df[c]
            ]
            for c in self._df.columns
        }


class NestedTemporalTable(FakeTable):
    """Real pykx .pa() fails on nested temporal columns; .py() works."""

    def pa(self) -> pa.Table:
        raise TypeError(
            "int() argument must be a string, a bytes-like object or a "
            "real number, not 'datetime.date'"
        )


class FakeVector:
    """Flat q vector stand-in; .pa() works (like real DateVector etc.)."""

    def __init__(self, array: Any) -> None:
        self._a = array

    def __len__(self) -> int:
        return len(self._a)

    def pa(self) -> pa.Array:
        return pa.array(self._a)

    def np(self) -> Any:
        return self._a


class FakeMonthVector(FakeVector):
    """Real MonthVector.pa() raises — datetime64[M] has no Arrow type."""

    def pa(self) -> pa.Array:
        raise TypeError("Unsupported numpy dtype: datetime64[M]")


class FakeList:
    """pykx.List stand-in: nested column, one vector per row; .pa() fails."""

    def __init__(self, vectors: list[FakeVector]) -> None:
        self._vectors = vectors

    def __iter__(self) -> Any:
        return iter(self._vectors)

    def pa(self) -> pa.Array:
        raise TypeError(
            "int() argument must be a string, a bytes-like object or a "
            "real number, not 'datetime.date'"
        )


class FakeColumnTable(FakeTable):
    """Table whose whole-table .pa() fails but exposes per-column access."""

    def __init__(self, cols: dict[str, Any]) -> None:
        self._cols = cols

    def pa(self) -> pa.Table:
        raise TypeError(
            "int() argument must be a string, a bytes-like object or a "
            "real number, not 'datetime.date'"
        )

    def __getitem__(self, name: str) -> Any:
        return self._cols[name]

    @property
    def columns(self) -> FakeK:
        return FakeK(list(self._cols))


class FakeKeyedTable:
    """pykx.KeyedTable stand-in: a pair of tables, .pa() unimplemented."""

    def __init__(self, keys: FakeTable, values: FakeTable) -> None:
        self._keys = keys
        self._values = values

    def keys(self) -> FakeTable:
        return self._keys

    def values(self) -> FakeTable:
        return self._values

    def pa(self) -> pa.Table:
        raise NotImplementedError  # mirrors real pykx.KeyedTable.pa()


class FakeK:
    def __init__(self, value: Any) -> None:
        self._value = value

    def py(self) -> Any:
        return self._value


class FakeQConnection:
    """Callable like a pykx connection; returns a canned result."""

    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[tuple[Any, ...]] = []
        self.closed = False

    def __call__(self, query: str, *params: Any) -> Any:
        self.calls.append((query, *params))
        return self.result

    def close(self) -> None:
        self.closed = True


class FakeSyncQConnection(FakeQConnection):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(FakeK(None))
        self.args = args
        self.kwargs = kwargs


@pytest.fixture(autouse=True)
def fake_pykx(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    module = types.ModuleType("pykx")
    module.Table = FakeTable  # type: ignore[attr-defined]
    module.List = FakeList  # type: ignore[attr-defined]
    module.KeyedTable = FakeKeyedTable  # type: ignore[attr-defined]
    module.SyncQConnection = FakeSyncQConnection  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pykx", module)
    return module


def run(result: Any, arrow_only: bool = True) -> pykx_dbapi.Cursor:
    conn = pykx_dbapi.Connection(FakeQConnection(result), arrow_only=arrow_only)
    return conn.cursor().execute("q")


def test_table_result() -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1.5, 2.5, 3.5]})
    cursor = run(FakeTable(df))
    assert [col[0] for col in cursor.description] == ["a", "b"]
    assert cursor.fetchall() == [(1, 1.5), (2, 2.5), (3, 3.5)]
    assert cursor.rowcount == 3


def test_table_result_is_arrow() -> None:
    df = pd.DataFrame({"a": [1, 2, 3]})
    table = run(FakeTable(df)).fetch_arrow_table()
    assert isinstance(table, pa.Table)
    assert table.column_names == ["a"]
    assert table.num_rows == 3


def test_keyed_table_raises_by_default() -> None:
    keyed = FakeKeyedTable(
        keys=FakeTable(pd.DataFrame({"k": [1, 2]})),
        values=FakeTable(pd.DataFrame({"v": [10, 20]})),
    )
    with pytest.raises(
        pykx_dbapi.NotSupportedError, match=r"unkey .* 0!|arrow_only=False"
    ):
        run(keyed)


def test_keyed_table_stitched_when_arrow_only_disabled() -> None:
    keyed = FakeKeyedTable(
        keys=FakeTable(pd.DataFrame({"k": [1, 2]})),
        values=FakeTable(pd.DataFrame({"v": [10, 20]})),
    )
    cursor = run(keyed, arrow_only=False)
    assert [col[0] for col in cursor.description] == ["k", "v"]
    assert cursor.fetchall() == [(1, 10), (2, 20)]


def test_null_result_has_no_result_set() -> None:
    cursor = run(FakeK(None))
    assert cursor.description is None
    assert cursor.rowcount == -1
    assert cursor.fetchall() == []
    with pytest.raises(pykx_dbapi.InterfaceError):
        cursor.fetch_arrow_table()


def test_atom_result() -> None:
    cursor = run(FakeK(42))
    assert [col[0] for col in cursor.description] == ["result"]
    assert cursor.fetchall() == [(42,)]


def test_vector_result() -> None:
    cursor = run(FakeK(["x", "y", "z"]))
    assert cursor.fetchall() == [("x",), ("y",), ("z",)]


def test_date_and_null_temporal_results() -> None:
    import datetime

    cursor = run(FakeK([datetime.date(2026, 7, 3), pd.NaT]))
    assert cursor.fetchall() == [(datetime.date(2026, 7, 3),), (None,)]
    cursor = run(FakeK(pd.NaT))  # 0Nd atom
    assert cursor.fetchall() == [(None,)]


def test_deprecated_z_datetime_gets_clear_error() -> None:
    class FakeZ:
        def py(self) -> Any:
            raise TypeError(
                "The q datetime type is deprecated, and can only be "
                "accessed with the keyword argument `raw=True`"
            )

    with pytest.raises(
        pykx_dbapi.NotSupportedError, match=r'cast to timestamp .* "p"\$x'
    ):
        run(FakeZ())


def test_nested_temporals_convert_in_arrow_only_mode() -> None:
    """select date, close by ticker (unkeyed with 0!) yields nested vector
    columns; whole-table .pa() fails, but the per-column retry builds Arrow
    list columns — including month/minute, whose numpy units Arrow lacks."""
    from datetime import date as datetime_date

    import numpy as np

    table = FakeColumnTable(
        {
            "d": FakeList(
                [
                    FakeVector(
                        np.array(["2026-07-01", "2026-07-02"], dtype="datetime64[D]")
                    ),
                    FakeVector(np.array(["2026-07-03"], dtype="datetime64[D]")),
                ]
            ),
            "m": FakeList(
                [
                    FakeMonthVector(
                        np.array(["2026-07", "2026-08"], dtype="datetime64[M]")
                    ),
                    FakeMonthVector(np.array(["2026-09"], dtype="datetime64[M]")),
                ]
            ),
            "v": FakeVector(np.array([1.5, 2.5])),
        }
    )
    result = run(table).fetch_arrow_table()  # arrow_only=True: must not raise
    assert str(result.schema.field("d").type) == "list<item: date32[day]>"
    assert str(result.schema.field("m").type) == "list<item: timestamp[ms]>"
    assert result["d"].to_pylist()[1] == [datetime_date(2026, 7, 3)]
    assert result["v"].to_pylist() == [1.5, 2.5]


def test_keyed_table_with_nested_date_vectors() -> None:
    """select date, close by ticker — non-aggregated columns come back as
    one vector per group; pykx's .pa() can't convert those, so with
    arrow_only=False the value side falls back to .py() while the key side
    stays Arrow-native."""
    import datetime

    import numpy as np

    values = pd.DataFrame(
        {
            "d": [
                np.array([datetime.date(2026, 7, 1), datetime.date(2026, 7, 2)]),
                np.array([datetime.date(2026, 7, 3)]),
            ],
            "v": [np.array([1.5, 2.5]), np.array([3.5])],
        }
    )
    keyed = FakeKeyedTable(
        keys=FakeTable(pd.DataFrame({"ticker": ["AAPL", "MSFT"]})),
        values=NestedTemporalTable(values),
    )
    cursor = run(keyed, arrow_only=False)
    assert [c[0] for c in cursor.description] == ["ticker", "d", "v"]
    table = cursor.fetch_arrow_table()
    assert str(table.schema.field("d").type) == "list<item: date32[day]>"
    rows = cursor.fetchall()
    assert rows[0][0] == "AAPL"
    assert rows[0][1] == [datetime.date(2026, 7, 1), datetime.date(2026, 7, 2)]


def test_unconvertible_table_raises_by_default() -> None:
    import datetime

    import numpy as np

    df = pd.DataFrame({"d": [np.array([datetime.date(2026, 7, 1)])]})
    with pytest.raises(
        pykx_dbapi.NotSupportedError, match=r"ungroup|arrow_only=False"
    ):
        run(NestedTemporalTable(df))


def test_unconvertible_table_falls_back_when_arrow_only_disabled() -> None:
    import datetime

    import numpy as np

    df = pd.DataFrame(
        {"d": [np.array([datetime.date(2026, 7, 1)])], "v": [np.array([1.5])]}
    )
    cursor = run(NestedTemporalTable(df), arrow_only=False)
    assert [c[0] for c in cursor.description] == ["d", "v"]
    assert cursor.fetchall() == [([datetime.date(2026, 7, 1)], [1.5])]


def test_mixed_list_falls_back_to_strings() -> None:
    cursor = run(FakeK([1, "a", 2.5]))
    assert cursor.fetchall() == [("1",), ("a",), ("2.5",)]


def test_dict_result() -> None:
    cursor = run(FakeK({"a": 1, "b": 2}))
    assert [col[0] for col in cursor.description] == ["key", "value"]
    assert cursor.fetchall() == [("a", 1), ("b", 2)]


def test_fetchone_and_fetchmany() -> None:
    cursor = run(FakeK([1, 2, 3, 4]))
    assert cursor.fetchone() == (1,)
    assert cursor.fetchmany(2) == [(2,), (3,)]
    assert cursor.fetchall() == [(4,)]
    assert cursor.fetchone() is None


def test_pep249_module_surface() -> None:
    assert pykx_dbapi.apilevel == "2.0"
    assert pykx_dbapi.paramstyle == "qmark"
    for name in (
        "Warning", "Error", "InterfaceError", "DatabaseError", "DataError",
        "OperationalError", "IntegrityError", "InternalError",
        "ProgrammingError", "NotSupportedError",
    ):
        assert isinstance(getattr(pykx_dbapi, name), type)
    assert issubclass(pykx_dbapi.NotSupportedError, pykx_dbapi.Error)


def test_executemany_arraysize_and_size_hints() -> None:
    q = FakeQConnection(FakeK(1))
    cursor = pykx_dbapi.Connection(q).cursor()
    cursor.executemany("{[t] count get t}", [("trades",), ("quotes",)])
    assert q.calls == [
        ("{[t] count get t}", "trades"),
        ("{[t] count get t}", "quotes"),
    ]
    cursor = run(FakeK([1, 2, 3]))
    cursor.arraysize = 2
    assert cursor.fetchmany() == [(1,), (2,)]  # defaults to arraysize
    cursor.setinputsizes([])
    cursor.setoutputsize(1)


def test_parameters_passed_through() -> None:
    q = FakeQConnection(FakeK(1))
    cursor = pykx_dbapi.Connection(q).cursor()
    cursor.execute("{[t] count get t}", ("trades",))
    assert q.calls == [("{[t] count get t}", "trades")]


def test_connect_disables_context_interface() -> None:
    """pykx's ctx interface sends k) code, which q-only servers reject."""
    conn = pykx_dbapi.connect(host="localhost", port=1337)
    assert conn._q.kwargs["no_ctx"] is True
    assert pykx_dbapi.connect(no_ctx=False)._q.kwargs["no_ctx"] is False


def test_connect_arrow_only_flag() -> None:
    conn = pykx_dbapi.connect(host="localhost", port=1337)
    assert conn._arrow_only is True
    assert "arrow_only" not in conn._q.kwargs  # not leaked to pykx
    assert pykx_dbapi.connect(arrow_only=False)._arrow_only is False


def test_connection_close_and_transactions() -> None:
    q = FakeQConnection(FakeK(None))
    conn = pykx_dbapi.Connection(q)
    conn.commit()
    conn.rollback()
    conn.close()
    assert q.closed


def test_adbc_get_objects() -> None:
    q = FakeQConnection(FakeK(["trades", "quotes"]))
    conn = pykx_dbapi.Connection(q)
    rows = conn.adbc_get_objects(depth="tables").read_all().to_pylist()
    assert rows == [
        {
            "catalog_name": "default",
            "catalog_db_schemas": [
                {
                    "db_schema_name": "",
                    "db_schema_tables": [
                        {"table_name": "trades", "table_type": "TABLE"},
                        {"table_name": "quotes", "table_type": "TABLE"},
                    ],
                }
            ],
        }
    ]
    assert q.calls == [("tables[]",)]

    catalogs_only = conn.adbc_get_objects(depth="catalogs").read_all()
    assert catalogs_only.to_pylist() == [{"catalog_name": "default"}]


def test_adbc_get_table_schema() -> None:
    # 0!meta comes back as columns c (names) and t (type chars); pykx
    # renders the char vector as bytes.
    meta = {"c": ["sym", "price", "size", "text"], "t": b"sfjC"}
    q = FakeQConnection(FakeK(meta))
    schema = pykx_dbapi.Connection(q).adbc_get_table_schema("trades")
    assert q.calls == [("{[t] 0!meta get t}", "trades")]
    assert schema.names == ["sym", "price", "size", "text"]
    assert [str(t) for t in schema.types] == [
        "string",
        "double",
        "int64",
        "string",
    ]


def test_arrow_type_mapping() -> None:
    import pyarrow as pa

    assert pykx_dbapi._arrow_type("p") == pa.timestamp("ns")
    assert pykx_dbapi._arrow_type(b"d") == pa.date32()
    assert pykx_dbapi._arrow_type(" ") == pa.null()  # nested/mixed column


def test_marimo_detects_connection_as_adbc() -> None:
    """The contract test: fails loudly if marimo's duck-typing changes."""
    marimo_adbc = pytest.importorskip("marimo._sql.engines.adbc")

    conn = pykx_dbapi.Connection(FakeQConnection(FakeK(None)))
    assert marimo_adbc.AdbcDBAPIEngine.is_compatible(conn)
    assert marimo_adbc.AdbcDBAPIEngine(conn).dialect == "kdb"

    from marimo._sql.get_engines import get_engines_from_variables

    engines = get_engines_from_variables([("conn", conn)])
    assert len(engines) == 1
    assert isinstance(engines[0][1], marimo_adbc.AdbcDBAPIEngine)


def test_polars_read_database_reads_via_dbapi() -> None:
    """Contract test: a second, unrelated consumer (polars.read_database)
    drives the connection end to end and gets correct data back.

    Note polars does NOT use our fetch_arrow_table here: its Arrow fast path
    is gated on a hardcoded driver-name allowlist (ARROW_DRIVER_REGISTRY,
    keyed on the connection's top-level module name), and "pykx_dbapi" isn't in
    it — unlike marimo, which picks the Arrow path by duck-typing. So polars
    reads through the plain DB-API row path (fetchall). This pins that the
    row path stays correct for non-marimo consumers."""
    pl = pytest.importorskip("polars")

    df = pd.DataFrame({"sym": ["AAPL", "MSFT"], "price": [1.0, 2.0]})
    conn = pykx_dbapi.Connection(FakeQConnection(FakeTable(df)))

    result = pl.read_database("select from trades", conn)

    assert isinstance(result, pl.DataFrame)
    assert result.columns == ["sym", "price"]
    assert result.to_dict(as_series=False) == {
        "sym": ["AAPL", "MSFT"],
        "price": [1.0, 2.0],
    }


def test_marimo_executes_query() -> None:
    """End-to-end through marimo's ADBC engine with a fake q connection."""
    marimo_adbc = pytest.importorskip("marimo._sql.engines.adbc")

    df = pd.DataFrame({"sym": ["AAPL", "MSFT"], "price": [1.0, 2.0]})
    conn = pykx_dbapi.Connection(FakeQConnection(FakeTable(df)))
    result = marimo_adbc.AdbcDBAPIEngine(conn).execute(
        "select from trades where sym=`AAPL"
    )
    assert result is not None
    assert list(result.columns) == ["sym", "price"]
