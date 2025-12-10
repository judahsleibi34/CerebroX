import pandas as pd 
import numpy as np 
import re 
import unicodedata
from pathlib import Path
from config import COLUMN_NAME_MAP, COLUMN_VALUE_MAP, normalize_column_name, columns_to_drop

class Data_analysis:

    @staticmethod
    def _normalize_header(s: str) -> str:
        if pd.isna(s):
            return s
        s = unicodedata.normalize("NFKC", str(s))
        s = s.replace("\u200f", "").replace("\u200e", "").replace("\u0640", "")  
        s = re.sub(r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED]", "", s)          
        s = re.sub(r"\s+", " ", s).strip()                                       
        s = (s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
            .replace("ى", "ي").replace("ئ", "ي").replace("ؤ", "و").replace("ة", "ه"))
        trans = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")                       
        s = s.translate(trans)
        return s

    @staticmethod
    def _normalize_value(s: str) -> str:
        if pd.isna(s):
            return s
        s = unicodedata.normalize("NFKC", str(s))
        s = s.replace("\u200f", "").replace("\u200e", "").replace("\u0640", "")
        s = re.sub(r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED]", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        s = (s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
             .replace("ى", "ي").replace("ئ", "ي").replace("ؤ", "و").replace("ة", "ه"))
        s = s.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
        return s

    def __init__(self, dataset_path_new: str):
        self.csv_path = Path(dataset_path_new)          
        self.default_dir = self.csv_path.parent         
        self.default_dir.mkdir(parents=True, exist_ok=True)
        self.original_name = self.csv_path.name         
        self.base_stem = self.csv_path.stem   

    def data_loading(self): 
        self.data_frame = pd.read_csv(self.csv_path)
        self.data_frame.columns = self.data_frame.columns.str.strip()
        self.data_frame.columns = self.data_frame.columns.str.replace('\n', ' ', regex=True)

        print('=' * 100)
        print("Data loaded successfully")
        print(f"Shape: {self.data_frame.shape}")
        print(f"Columns: {len(self.data_frame.columns)}")
        print('=' * 100)

    def normalize_headers(self):
        original_cols = self.data_frame.columns.tolist()
        self.data_frame.columns = [self._normalize_header(c) for c in self.data_frame.columns]
        
        print("\n=== HEADER NORMALIZATION ===")
        for orig, norm in zip(original_cols, self.data_frame.columns):
            if orig != norm:
                print(f"  '{orig}' -> '{norm}'")

    def column_rename(self):
        print(f"\n{'='*100}")
        print("STARTING COLUMN RENAME PROCESS")
        print(f"{'='*100}\n")
        
        original_columns = self.data_frame.columns.tolist()
        print(f"Original columns ({len(original_columns)}):")
        for i, col in enumerate(original_columns, 1):
            print(f"  {i}. '{col}'")
        
        print(f"\n=== HEADER NORMALIZATION ===")
        normalized_columns = {}
        normalization_changes = []
        
        for col in self.data_frame.columns:
            normalized = normalize_column_name(col)
            normalized_columns[col] = normalized
            
            if col != normalized:
                normalization_changes.append((col, normalized))
                print(f"  '{col}' -> '{normalized}'")
        
        if not normalization_changes:
            print("  No normalization changes needed.")
        
        self.data_frame.rename(columns=normalized_columns, inplace=True)
        
        print(f"\n=== COLUMN MAPPING PROCESS ===")
        print(f"Available mappings in COLUMN_NAME_MAP: {len(COLUMN_NAME_MAP)}\n")
        
        rename_map = {}
        mapped_count = 0
        unmapped_columns = []
        
        for old_col in self.data_frame.columns:
            if old_col in COLUMN_NAME_MAP:
                new_col = COLUMN_NAME_MAP[old_col]
                rename_map[old_col] = new_col
                mapped_count += 1
                print(f"✓ Column '{old_col}' -> '{new_col}'")
            else:
                unmapped_columns.append(old_col)
                print(f"✗ Column '{old_col}' -> NO MAPPING FOUND")
        
        # Apply the rename map
        if rename_map:
            self.data_frame.rename(columns=rename_map, inplace=True)
            print(f"\n✓ Successfully renamed {mapped_count} columns")
        else:
            print(f"\n✗ WARNING: NO COLUMNS WERE RENAMED!")
        
        # Warning for unmapped columns
        if unmapped_columns:
            print(f"\n{'!'*100}")
            print(f"WARNING: {len(unmapped_columns)} columns have NO mapping:")
            for col in unmapped_columns:
                print(f"  - '{col}'")
            print(f"{'!'*100}\n")
        
        # Final summary
        print(f"\n{'='*100}")
        print("FINAL COLUMN NAMES AFTER RENAMING:")
        print(f"{'='*100}")
        final_columns = self.data_frame.columns.tolist()
        for i, col in enumerate(final_columns, 1):
            if i <= len(original_columns) and original_columns[i-1] != col:
                print(f"  {i}. '{col}' (was: '{original_columns[i-1]}')")
            else:
                print(f"  {i}. '{col}' (unchanged)")
        
        # Save the renamed dataframe
        renamed_path = self.default_dir / f"{self.base_stem}_renamed.csv"
        self.data_frame.to_csv(renamed_path, index=False, encoding='utf-8-sig')
        print(f"\n✓ Renamed DataFrame saved as: {renamed_path}")
        print(f"{'='*100}\n")

    def data_investigation(self):
        print("\n" + "=" * 100)
        print("DATA INVESTIGATION")
        print("=" * 100)
        
        self.data_frame.info()
        self.renamed_columns = self.data_frame.columns.tolist()
         
        for column in self.renamed_columns: 
            Column = self.data_frame[column]
            print(f"\n--- Column: '{column}' ---")
            
            if pd.api.types.is_numeric_dtype(Column):
                print(f"Type: Numeric")
                print(f"Missing: {Column.isna().sum()}")
                print(f"Duplicates: {Column.duplicated().sum()}")
                print("Statistics:")
                print(Column.describe())
                
                if pd.api.types.is_integer_dtype(Column):
                    self.data_frame[column] = pd.to_numeric(Column, downcast='integer')
                elif pd.api.types.is_float_dtype(Column):
                    self.data_frame[column] = pd.to_numeric(Column, downcast='float')  
                    
            elif pd.api.types.is_datetime64_any_dtype(Column):
                print(f"Type: Datetime")
                print(f"Missing: {Column.isna().sum()}")

            elif pd.api.types.is_bool_dtype(Column) or pd.api.types.is_object_dtype(Column):
                print(f"Type: Categorical/Object")
                print(f"Unique values: {len(Column.unique())}")
                print(f"Duplicates: {Column.duplicated().sum()}")
                print(f"Missing: {Column.isna().sum()}")

                uniques = Column.dropna().unique()
                if len(uniques) <= 30:
                    print("All unique values:")
                    for val in uniques:
                        print(f"  - {val}")
                else:
                    print(f"Unique values (first 20 of {len(uniques)}):")
                    for val in uniques[:20]:
                        print(f"  - {val}")

            print("-" * 60)

    def data_cleaning(self): 
        print("\n" + "=" * 100)
        print("DATA CLEANING PROCESS")
        print("=" * 100)
        
        print("\nDataFrame info before cleaning:")
        print(self.data_frame.info())

        existing_drops = [col for col in columns_to_drop if col in self.data_frame.columns]
        if existing_drops:
            self.data_frame = self.data_frame.drop(columns=existing_drops)
            print(f"\n✓ Dropped columns: {existing_drops}")
        
        obj_cols = self.data_frame.select_dtypes(include="object").columns
        print(f"\nNormalizing {len(obj_cols)} object columns...")
        self.data_frame[obj_cols] = self.data_frame[obj_cols].apply(
            lambda col: col.map(self._normalize_value)
        )

        print("\n=== APPLYING VALUE MAPPINGS ===")
        print(f"Value mappings defined for {len(COLUMN_VALUE_MAP)} columns")
        
        successfully_mapped = 0
        for new_col_name, value_mapping in COLUMN_VALUE_MAP.items():
            if new_col_name in self.data_frame.columns:
                print(f"\n✓ Column '{new_col_name}' found")
                print(f"  Applying {len(value_mapping)} value mappings...")
                
                norm_value_mapping = {
                    self._normalize_value(k): v 
                    for k, v in value_mapping.items()
                }

                original_values = self.data_frame[new_col_name].dropna().unique()
                
                print(f"  Original unique values ({len(original_values)}):")
                for val in original_values[:10]:
                    print(f"    - '{val}'")
                
                self.data_frame[new_col_name] = self.data_frame[new_col_name].replace(
                    norm_value_mapping
                )
                successfully_mapped += 1
                
                new_values = self.data_frame[new_col_name].dropna().unique()
                print(f"  New unique values ({len(new_values)}):")
                for val in new_values[:10]:
                    print(f"    - '{val}'")
            else:
                print(f"\n✗ Column '{new_col_name}' NOT FOUND in DataFrame")
                print(f"  Available columns containing similar text:")
                for col in self.data_frame.columns:
                    if any(word.lower() in col.lower() for word in new_col_name.split()[:2]):
                        print(f"    - '{col}'")

        print(f"\n✓ Successfully applied value mappings to {successfully_mapped} columns")
        print("=" * 100)

        initial_rows = len(self.data_frame)
        self.data_frame.drop_duplicates(inplace=True)
        removed_rows = initial_rows - len(self.data_frame)
        print(f"\n✓ Removed {removed_rows} duplicate rows")

        print("\n=== MISSING VALUES SUMMARY ===")
        missing = self.data_frame.isna().sum()
        missing = missing[missing > 0]
        if len(missing) > 0:
            print(missing)
        else:
            print("No missing values!")

        print("\n" + "=" * 100)
        cleaned_path = self.default_dir / f"{self.base_stem}_cleaned.csv"
        self.data_frame.to_csv(cleaned_path, index=False, encoding='utf-8-sig')
        print(f"✓ Cleaned dataframe saved as: {cleaned_path}")
        print("=" * 100)

        print("\nCleaned dataframe info:")
        print(self.data_frame.info())
        print("=" * 100)
