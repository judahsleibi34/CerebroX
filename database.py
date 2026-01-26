import os
import json
import re
import hashlib
from datetime import datetime
from pathlib import Path

import pandas as pd
import pymysql
from sqlalchemy import create_engine, inspect
from sqlalchemy.types import Text, BigInteger, Float, DateTime
from urllib.parse import quote_plus
import sys


def get_app_root():
    if hasattr(sys, "_MEIPASS"):
        return Path(sys.executable).parent
    else:
        return Path(__file__).parent


def get_log_file():
    root = get_app_root()
    logs_dir = root / "logs"
    logs_dir.mkdir(exist_ok=True)

    log_path = logs_dir / "cerebrox_history.json"

    if not log_path.exists():
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump([], f, indent=4)

    return log_path


class CerebroXDB:
    def __init__(self,
                 host="localhost",
                 user="root",
                 password="Jus@12345",
                 port=3306,
                 database="cerebrox_data"):
        
        self.host = host
        self.user = user
        self.password = password
        self.port = port
        self.database = database

        self._ensure_database_exists()

        encoded_pw = quote_plus(password)
        self.engine = create_engine(
            f"mysql+pymysql://{user}:{encoded_pw}@{host}:{port}/{database}?charset=utf8mb4"
        )

    def _ensure_database_exists(self):
        conn = pymysql.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            port=self.port
        )
        with conn.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{self.database}` "
                           "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        conn.commit()
        conn.close()

    def sanitize(self, name: str) -> str:
        name = str(name).lower()
        name = re.sub(r"[^a-zA-Z0-9_]+", "_", name)
        return name.strip("_")[:60]

    def get_next_version(self, base_name: str) -> int:
        inspector = inspect(self.engine)
        tables = inspector.get_table_names()

        version = 0
        pattern = re.compile(rf"^{base_name}.*_v(\d+)$")

        for table in tables:
            match = pattern.match(table)
            if match:
                v = int(match.group(1))
                version = max(version, v)

        return version + 1

    def truncate_column_names(self, df: pd.DataFrame, max_length=64) -> pd.DataFrame:
        """
        Truncate column names to MySQL's 64-character identifier limit.
        Adds a hash suffix to ensure uniqueness.
        """
        df2 = df.copy()
        new_columns = []
        
        for col in df2.columns:
            col_str = str(col)
            if len(col_str) > max_length:
                # Create hash for uniqueness
                hash_suffix = hashlib.sha1(col_str.encode()).hexdigest()[:6]
                # Truncate and add hash
                truncated = col_str[:max_length - 7]  # Leave room for _hash
                new_name = f"{truncated}_{hash_suffix}"
                print(f"[DB] Column truncated: '{col_str[:50]}...' -> '{new_name}'")
                new_columns.append(new_name)
            else:
                new_columns.append(col_str)
        
        df2.columns = new_columns
        return df2

    def save_table(self, df: pd.DataFrame, table_name: str):
        if df.empty:
            raise ValueError(f"Cannot save empty DataFrame to {table_name}")

        df2 = df.copy()

        # ✅ STEP 1: Truncate long column names FIRST (MySQL 64-char limit)
        df2 = self.truncate_column_names(df2, max_length=64)

        # ✅ STEP 2: Handle duplicate columns
        if df2.columns.duplicated().any():
            dupes = df2.columns[df2.columns.duplicated()].tolist()
            print("[DB] Duplicate columns detected:", dupes)

            seen = {}
            new_cols = []
            for c in df2.columns:
                c = str(c)
                if c not in seen:
                    seen[c] = 0
                    new_cols.append(c)
                else:
                    seen[c] += 1
                    new_cols.append(f"{c}__{seen[c]}")
            df2.columns = new_cols

        # ✅ STEP 3: Convert object columns to string
        for col in df2.select_dtypes(include=["object"]).columns:
            df2[col] = df2[col].astype(str)

        # ✅ STEP 4: Map dtypes for SQL
        dtype_map = {}
        for col in df2.columns:
            dtype = str(df2[col].dtype)
            if "int" in dtype:
                dtype_map[col] = BigInteger
            elif "float" in dtype:
                dtype_map[col] = Float
            elif "datetime" in dtype:
                dtype_map[col] = DateTime
            else:
                dtype_map[col] = Text

        # ✅ STEP 5: Save to database
        df2.to_sql(
            table_name,
            self.engine,
            if_exists="replace",
            index=False,
            dtype=dtype_map,
            method="multi",
            chunksize=1000
        )
        print(f"[DB] ✅ Table '{table_name}' saved successfully with {len(df2)} rows and {len(df2.columns)} columns")

    def save_run(self, filename: str, merged_df: pd.DataFrame, sheets: dict):
        clean_file = self.sanitize(Path(filename).stem)
        base_name = f"cleaned_{clean_file}"

        version = self.get_next_version(base_name)

        created_tables = []

        # Save merged dataframe
        if merged_df is not None:
            raw_name = f"{base_name}_merged_v{version}"
            table_name = self._shorten_name(raw_name)

            print(f"[DB] Saving merged dataframe as table: {table_name}")
            self.save_table(merged_df, table_name)
            created_tables.append(table_name)

        # Save individual sheets
        for sheet_name, df in sheets.items():
            if df is None or df.empty:
                continue

            safe_sheet = self.sanitize(sheet_name)

            raw_sheet_name = f"{base_name}_{safe_sheet}_v{version}"
            sheet_table = self._shorten_name(raw_sheet_name)  
            self.save_table(df, sheet_table)
            created_tables.append(sheet_table)

        self._write_log(filename, version, created_tables)

        return version, created_tables

    def log(self, msg):
        try:
            print("[DB LOG]", msg)
        except:
            pass

    def _write_log(self, filename: str, version: int, tables: list):
        log_path = get_log_file()

        with open(log_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename": filename,
            "version": version,
            "tables": tables
        }

        data.append(entry)

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def _shorten_name(self, name: str, max_length=60) -> str:
        """Shorten table names to avoid MySQL limits"""
        name = name.lower().replace(" ", "_")
        if len(name) <= max_length:
            return name

        hash_suffix = hashlib.sha1(name.encode()).hexdigest()[:8]
        prefix = name[:max_length - 9]  
        return f"{prefix}_{hash_suffix}"