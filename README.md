# pykx-dbapi

A [PEP 249 (DB-API 2.0)](https://peps.python.org/pep-0249/) wrapper around
[pykx](https://code.kx.com/pykx/), that gives [marimo](https://marimo.io) notebooks
the ability to query kdb+/q servers.

It wraps a `pykx.SyncQConnection` in the standard `connect()` /
`Connection` / `Cursor` interface, plus the
[ADBC](https://arrow.apache.org/adbc/) extension methods marimo consumes:
query results flow as **Arrow** (near zero-copy into polars), and marimo's
datasources panel can browse tables and column types via q's `tables[]` and
`meta`. marimo auto-detects the connection — no plugins or configuration.
Queries are **q/qSQL text sent to the server verbatim**; there is no SQL
translation.

pykx's unlicensed mode is sufficient — IPC-only use needs no kdb+ license on
the client.

## Screenshot

![PyKX in Marimo](https://raw.githubusercontent.com/jparmstrong/pykx-dbapi/main/marimo_screenshot.png)

## Adding to a marimo notebook project

**uv-managed project** (marimo installed in a project venv):

```bash
uv add pykx-dbapi
uv run marimo edit notebook.py
```

**Plain pip:**

```bash
pip install pykx-dbapi
```

## Usage

In a marimo cell:

```python
import pykx_dbapi

conn = pykx_dbapi.connect(host="localhost", port=5001)
```

marimo detects `conn` as a database connection. Create a SQL cell, pick `conn`
as its engine, and write q or qSQL:

```q
select avg price by sym from trades where size > 100
```

Results come back as dataframes. Non-table results are shaped into one:
atoms/vectors as a single `result` column, dicts as `key`/`value` columns.
Statements returning q's generic null (e.g. assignments like `t:([] a:1 2 3)`)
produce no result set.

`connect(...)` passes its arguments straight to `pykx.SyncQConnection`, so
TLS, auth, and timeouts all work:
`connect(host, port, username=..., password=...)`. You can also wrap an
existing pykx connection: `pykx_dbapi.Connection(existing)`.

Works outside marimo too — anything DB-API-aware can use it, e.g.
`pandas.read_sql("select from trades", conn)` or `polars.read_database(...)`.
Errors raised by the wrapper follow the PEP 249 hierarchy
(`pykx_dbapi.Error` and subclasses).

### Arrow-only mode

By default every table result must convert to Arrow (`arrow_only=True`).
Nested vector columns (a select-by without aggregation, e.g.
`0!select date, close by ticker`) convert to Arrow list columns — including
temporals: nested dates land as `list<date32>`; months and minutes, which
Arrow has no native unit for, are cast to `timestamp[ms]`/`duration[ms]`.

Results pykx cannot convert raise `pykx_dbapi.NotSupportedError` telling you
how to reshape — notably **keyed tables** (`select ... by ...`): unkey in the
query with `0!select ...`.

`connect(..., arrow_only=False)` instead stitches keyed tables from their
key/value parts and falls back to a Python-object conversion for tables pykx
can't handle (slower; timestamps land at µs rather than ns precision).

### q braces vs. marimo SQL cells

marimo compiles SQL cells to Python **f-strings** (that's how `{python_var}`
interpolation works), so a literal `{...}` — a q lambda — is evaluated as a
Python expression and typically fails with a `NameError`. Double the braces
in SQL cells:

```q
0!{{P:asc distinct x`ticker; exec P#(ticker!close) by date:date from x}} ungroup ...
```

Or run lambda-heavy queries from a Python cell, where no interpolation
happens: `mo.sql("0!{...} ...", engine=conn)`.

## Caveats

- marimo treats `kdb` as a remote dialect and won't introspect eagerly. To
  populate the datasources panel automatically, enable discovery in marimo's
  config (`datasources: auto_discover_schemas/tables/columns`).
- marimo's optional SQL-validation feature prefixes queries with `EXPLAIN`,
  which q rejects. Harmless — it surfaces as a validation warning; leave
  validation off for this connection.
- `commit()`/`rollback()` are no-ops: q IPC has no transactions.
- The deprecated q datetime (`z`) type is not supported (pykx refuses to
  convert it); cast to timestamp in your query, e.g. `"p"$x`.
- pykx's context interface is disabled by default (`no_ctx=True`): this
  wrapper never uses it, and its k)-dialect bootstrap breaks some q-only
  servers (e.g. KX Insights). Pass `connect(..., no_ctx=False)` to re-enable.

## Development

```bash
uv sync
uv run pytest
```

Tests run against a fake pykx — no q server needed. They include contract
tests that fail loudly if marimo's or polars' DB-API duck-typing ever changes.
