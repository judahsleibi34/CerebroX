import sqlite3
from pathlib import Path
from typing import Optional
import pandas as pd


class CerebroXDB:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._init_schema()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_schema(self):
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS datasets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    table_name TEXT NOT NULL UNIQUE,
                    source_csv TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

    def save_snapshot(
        self,
        df: pd.DataFrame,
        name: str,
        source_csv: Optional[Path] = None
    ) -> str:
        table_name = name.replace(" ", "_").lower()

        with self._connect() as conn:
            df.to_sql(table_name, conn, if_exists="replace", index=False)

            conn.execute(
                """
                INSERT INTO datasets (name, table_name, source_csv)
                VALUES (?, ?, ?)
                ON CONFLICT(table_name)
                DO UPDATE SET name = excluded.name,
                              source_csv = excluded.source_csv
                """,
                (name, table_name, str(source_csv) if source_csv else None)
            )

        return table_name

    def list_datasets(self) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query("SELECT * FROM datasets ORDER BY created_at DESC", conn)

    def load_dataset(self, table_name: str) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

    def run_query(self, sql: str, params=None) -> pd.DataFrame:
        params = params or []
        with self._connect() as conn:
            return pd.read_sql_query(sql, conn, params=params)