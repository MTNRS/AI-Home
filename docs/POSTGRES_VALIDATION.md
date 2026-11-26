# PostgreSQL validation on Windows

A dedicated portable PostgreSQL 17.12 cluster was created under the ignored
`artifacts/postgres-validation` directory. It listened only on 127.0.0.1, used
SCRAM authentication with a generated test credential and a separate
`aihome_validation` database. No Windows service was installed and no existing
database was used. The cluster is stopped after validation.

Binary source: the PostgreSQL Windows page links to EDB's binary distribution:
https://www.postgresql.org/download/windows/
https://www.enterprisedb.com/download-postgresql-binaries

Archive: `postgresql-17.12-2-windows-x64-binaries.zip`, 386262342 bytes.
Verified against the publisher's `.zip.sha256` sidecar:
`520c8c6f19abb3b31fa83ec5e486e2f769eedd7593d94e60a7cda8a5bfb4df91`.
Only pgsql/bin, pgsql/lib and pgsql/share were extracted.

## Reproduce against a dedicated test database

Set `AI_HOME_TEST_POSTGRES_URL` to a psycopg SQLAlchemy URL for a disposable,
empty PostgreSQL database. Do not use production: tests migrate its schema and
create synthetic accounts and ledger transactions. Then run:

```powershell
.venv-cuda\Scripts\python.exe -m pytest services/api/tests/test_postgres_integration.py -q --tb=short
.venv-cuda\Scripts\python.exe -m pytest -q --tb=short
```

The fixture now binds Alembic's `AI_HOME_DATABASE_URL` to the test URL for the
fixture lifetime, restoring the prior environment afterward. Previously the
fixture could migrate a different configured database while testing this URL.

For portable Windows startup, use initdb with a private password file and
SCRAM authentication; configure listen_addresses=127.0.0.1 and a free port.
Launch pg_ctl with CREATE_NO_WINDOW, log output to a file and avoid captured
pipes inherited by the persistent server. Stop with `pg_ctl -D <data> -m fast
-w stop` after the test process exits. Keep credentials and data out of git.

## Finding from actual PostgreSQL execution

The first real run passed the four existing migration/identity/scheduler/ledger
tests but failed the new duplicate-inference reservation test: both concurrent
callers reported a newly created reservation. The driver did not provide a
reliable insert rowcount for the conflict path. Ledger idempotency prevented a
second debit, but the admission result could authorize duplicate computation.

Reservation insertion and terminal resolution now use RETURNING and inspect the
returned request ID. Account-existence write locks use RETURNING too. Tests
cover concurrent same-key insertion, concurrent settlement, deadline fencing
and expired refund in PostgreSQL, in addition to SQLite coverage. This is local
single-server validation, not a multi-host or production load benchmark.

Final result: **263 passed, zero skipped**, including all five PostgreSQL tests.
The portable server was stopped and pg_ctl status confirmed no running server.
