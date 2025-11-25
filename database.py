import pymysql
import hashlib
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus
import re


class CerebroXDB:
    def __init__(self, host: str, user: str, password: str, port: int = 3306):
        self.host = host
        self.user = user
        self.password = password
        self.port = port
        self.registry_db = "cerebrox_data"
        
        self._ensure_registry_exists()
        
        encoded_password = quote_plus(password)
        self.registry_engine = create_engine(
            f"mysql+pymysql://{user}:{encoded_password}@{host}:{port}/{self.registry_db}"
        )
        
        self.current_project_id = None
        self.current_project_db = None
        self.current_engine = None


    def _ensure_registry_exists(self):
        conn = pymysql.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            port=self.port
        )
        
        try:
            with conn.cursor() as cursor:
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.registry_db}")
                conn.commit()
                
            conn.select_db(self.registry_db)
            
            with conn.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS projects (
                        project_id VARCHAR(50) PRIMARY KEY,
                        project_name VARCHAR(255) NOT NULL,
                        database_name VARCHAR(100) NOT NULL UNIQUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        source_files TEXT,
                        description TEXT,
                        status ENUM('active', 'archived', 'deleted') DEFAULT 'active'
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS project_datasets (
                        dataset_id INT AUTO_INCREMENT PRIMARY KEY,
                        project_id VARCHAR(50),
                        dataset_name VARCHAR(255) NOT NULL,
                        table_name VARCHAR(255) NOT NULL,
                        stage ENUM('raw', 'cleaned', 'merged', 'final') DEFAULT 'raw',
                        `row_count` INT,
                        `column_count` INT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (project_id) REFERENCES projects(project_id) ON DELETE CASCADE
                    )
                """)
                conn.commit()
        finally:
            conn.close()


    def _sanitize_name(self, name: str) -> str:
        name = re.sub(r'[^\w\s-]', '', name.lower())
        name = re.sub(r'[-\s]+', '_', name)
        return name[:50]


    def _sanitize_column_name(self, name: str) -> str:
        cleaned = str(name).strip()
        cleaned = re.sub(r'[^\w\s\u0600-\u06FF-]', '', cleaned)
        cleaned = re.sub(r'\s+', '_', cleaned)
        
        if len(cleaned.encode('utf-8')) > 60:
            hash_suffix = hashlib.md5(cleaned.encode()).hexdigest()[:8]
            max_length = 51
            cleaned = cleaned[:max_length] + '_' + hash_suffix
        
        return cleaned if cleaned else f"col_{hashlib.md5(str(name).encode()).hexdigest()[:8]}"


    def _generate_project_id(self, project_name: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        hash_suffix = hashlib.md5(project_name.encode()).hexdigest()[:6]
        return f"proj_{timestamp}_{hash_suffix}"


    def create_project(self, project_name: str, source_files: List[str] = None, description: str = None) -> str:
        project_id = self._generate_project_id(project_name)
        safe_name = self._sanitize_name(project_name)
        db_name = f"cerebrox_proj_{safe_name}_{project_id[-6:]}"
        
        conn = pymysql.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            port=self.port
        )
        
        try:
            with conn.cursor() as cursor:
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
                conn.commit()
            
            conn.select_db(self.registry_db)
            
            with conn.cursor() as cursor:
                source_files_str = ",".join(source_files) if source_files else None
                cursor.execute("""
                    INSERT INTO projects (project_id, project_name, database_name, source_files, description)
                    VALUES (%s, %s, %s, %s, %s)
                """, (project_id, project_name, db_name, source_files_str, description))
                conn.commit()
            
            print(f"✓ Created project '{project_name}' with ID: {project_id}")
            print(f"✓ Database: {db_name}")
            
            self.current_project_id = project_id
            self.current_project_db = db_name
            
            encoded_password = quote_plus(self.password)
            self.current_engine = create_engine(
                f"mysql+pymysql://{self.user}:{encoded_password}@{self.host}:{self.port}/{db_name}"
            )
            
            return project_id
            
        finally:
            conn.close()


    def load_project(self, project_id: str):
        query = "SELECT database_name FROM projects WHERE project_id = %s AND status = 'active'"
        result = pd.read_sql_query(query, self.registry_engine, params=(project_id,))
        
        if result.empty:
            raise ValueError(f"Project '{project_id}' not found or inactive")
        
        db_name = result.iloc[0]['database_name']
        
        self.current_project_id = project_id
        self.current_project_db = db_name
        
        encoded_password = quote_plus(self.password)
        self.current_engine = create_engine(
            f"mysql+pymysql://{self.user}:{encoded_password}@{self.host}:{self.port}/{db_name}"
        )
        
        print(f"✓ Loaded project: {project_id} → {db_name}")


    def save_dataset(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        stage: str = 'raw',
        source_csv: Optional[Path] = None
    ) -> str:
        if not self.current_project_id or not self.current_engine:
            raise ValueError("No project loaded. Call create_project() or load_project() first.")
        
        df_copy = df.copy()
        
        column_mapping = {}
        for col in df_copy.columns:
            sanitized = self._sanitize_column_name(col)
            column_mapping[col] = sanitized
        
        df_copy.rename(columns=column_mapping, inplace=True)
    
        safe_name = self._sanitize_name(dataset_name)
        date_str = datetime.now().strftime("%Y_%m_%d")
        cycle = 1
        table_name = f"{stage}_{safe_name}_cycle_{cycle}_{date_str}"
            
        df_copy.to_sql(table_name, self.current_engine, if_exists="replace", index=False)
        
        conn = pymysql.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.registry_db,
            port=self.port
        )
        
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO project_datasets 
                    (project_id, dataset_name, table_name, stage, row_count, column_count)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    self.current_project_id,
                    dataset_name,
                    table_name,
                    stage,
                    len(df),
                    len(df.columns)
                ))
                conn.commit()
        finally:
            conn.close()
        
        print(f"✓ Saved dataset '{dataset_name}' as '{table_name}' (stage: {stage})")
        return table_name
    

    def list_projects(self) -> pd.DataFrame:
        query = "SELECT * FROM projects WHERE status = 'active' ORDER BY last_modified DESC"
        return pd.read_sql_query(query, self.registry_engine)


    def list_datasets(self, project_id: str = None) -> pd.DataFrame:
        if project_id is None:
            project_id = self.current_project_id
        
        if not project_id:
            raise ValueError("No project specified")
        
        query = """
            SELECT * FROM project_datasets 
            WHERE project_id = %s 
            ORDER BY created_at DESC
        """
        return pd.read_sql_query(query, self.registry_engine, params=(project_id,))


    def load_dataset(self, table_name: str) -> pd.DataFrame:
        if not self.current_engine:
            raise ValueError("No project loaded")
        
        return pd.read_sql_query(f"SELECT * FROM {table_name}", self.current_engine)


    def archive_project(self, project_id: str):
        conn = pymysql.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.registry_db,
            port=self.port
        )
        
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "UPDATE projects SET status = 'archived' WHERE project_id = %s",
                    (project_id,)
                )
                conn.commit()
            print(f"✓ Archived project: {project_id}")
        finally:
            conn.close()


    def delete_project(self, project_id: str, permanent: bool = False):
        query = "SELECT database_name FROM projects WHERE project_id = %s"
        result = pd.read_sql_query(query, self.registry_engine, params=(project_id,))
        
        if result.empty:
            raise ValueError(f"Project '{project_id}' not found")
        
        db_name = result.iloc[0]['database_name']
        
        conn = pymysql.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            port=self.port
        )
        
        try:
            if permanent:
                with conn.cursor() as cursor:
                    cursor.execute(f"DROP DATABASE IF EXISTS {db_name}")
                    conn.commit()
                
                conn.select_db(self.registry_db)
                with conn.cursor() as cursor:
                    cursor.execute("DELETE FROM projects WHERE project_id = %s", (project_id,))
                    conn.commit()
                
                print(f"✓ Permanently deleted project: {project_id}")
            else:
                conn.select_db(self.registry_db)
                with conn.cursor() as cursor:
                    cursor.execute(
                        "UPDATE projects SET status = 'deleted' WHERE project_id = %s",
                        (project_id,)
                    )
                    conn.commit()
                print(f"✓ Marked project as deleted: {project_id}")
        finally:
            conn.close()