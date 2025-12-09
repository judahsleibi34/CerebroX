CREATE DATABASE if not exists cerebrox_data;

USE cerebrox_data;

CREATE TABLE IF NOT EXISTS projects (
    project_id VARCHAR(50) PRIMARY KEY,
    project_name VARCHAR(255) NOT NULL,
    database_name VARCHAR(100) NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    source_files TEXT,
    description TEXT,
    status ENUM('active', 'archived', 'deleted') DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS project_datasets (
    dataset_id INT AUTO_INCREMENT PRIMARY KEY,
    project_id VARCHAR(50),
    dataset_name VARCHAR(255) NOT NULL,
    table_name VARCHAR(255) NOT NULL,
    stage ENUM('raw', 'cleaned', 'merged', 'final') DEFAULT 'raw',
    row_count INT,
    column_count INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (project_id)
        REFERENCES projects (project_id)
        ON DELETE CASCADE
);

SELECT * FROM cerebrox_data.projects;
