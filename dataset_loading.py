import pandas as pd
import numpy as np
from config import sheets
from pathlib import Path

class Data_set: 
    def __init__(self, dataset_path_old: str, dataset_path_new: str):
        self.dataset_path_old = dataset_path_old
        self.dataset_path_new = Path(dataset_path_new)
        self.output_dir = self.dataset_path_new.parent
        self.base_name = self.dataset_path_new.stem.replace("_processed", "")
        self.sheets_dfs = {}      
        self.data_frame = None 

    def load_google_excel(self, file_path):
        df_raw = pd.read_excel(file_path, header=None, dtype=str)
        header_row_idx = df_raw.apply(lambda row: row.notna().sum(), axis=1).idxmax()
        df = df_raw.copy()
        df.columns = df.iloc[header_row_idx]
        df = df[header_row_idx + 1:].reset_index(drop=True)
        df = df.dropna(axis=1, how='all')
        df.columns = (
            df.columns.astype(str)
            .str.replace("\n", " ")
            .str.replace("  ", " ")
            .str.strip()
        )
        return df

    def _clean_piece(self, s):
        if pd.isna(s):
            return np.nan
        s = str(s).strip()
        if not s or s.lower().startswith("unnamed"):
            return np.nan
        return s

    def _choose_sep(self, tokens):
        candidates = [" / ", " - ", " | ", " · ", " → ", " :: "]
        for cand in candidates:
            if all(cand not in t for t in tokens):
                return cand
        return " | "

    def _score_depth(self, sheet, n):
        try:
            df_try = pd.read_excel(self.dataset_path_old, sheet_name=sheet, header=list(range(n)))
        except Exception:
            return -1, None, None
        levels = []
        for i in range(n):
            lvl = pd.Series([col[i] if len(col) > i else np.nan for col in df_try.columns]).map(self._clean_piece)
            levels.append(lvl.ffill())
        score = 0
        for j in range(len(df_try.columns)):
            if any(pd.notna(levels[i].iloc[j]) for i in range(n)):
                score += 1
        return score, levels, df_try

    def _auto_depth(self, sheet, max_try=8):
        best = (-1, None, None, None)
        for n in range(2, max_try + 1):
            sc, lvls, df_try = self._score_depth(sheet, n)
            if sc > best[0]:
                best = (sc, n, lvls, df_try)
        if best[1] is None:
            df = pd.read_excel(self.dataset_path_old, sheet_name=sheet, header=0)
            n = 1
            levels = [pd.Series(df.columns).map(self._clean_piece).ffill()]
            return n, levels, df
        return best[1], best[2], best[3]

    def load_excel_with_multilevel_header(self):
        excel_file = pd.ExcelFile(self.dataset_path_old)
        sheet_names = excel_file.sheet_names

        print("=" * 100)
        print(f"File loaded. Found {len(sheet_names)} sheets: {sheet_names}")

        frames = []
        self.sheets_dfs = {}   

        if len(sheets) != len(sheet_names):
            print(f"⚠ Warning: Configured sheets ({len(sheets)}) "
                f"and Excel sheets ({len(sheet_names)}) do not match.")
            print(f"Using Excel sheet names instead for unmatched entries.")

        for i, sheet in enumerate(sheet_names):
            print("-" * 60)
            print(f"Processing sheet: {sheet}")

            n, levels, df = self._auto_depth(sheet)

            tokens = set()
            for level in levels:
                tokens.update(x for x in level.dropna().astype(str).tolist())
            sep = self._choose_sep(tokens)

            final_cols, seen = [], {}

            for a in zip(*[lvl for lvl in levels]):
                parts = [p for p in a if pd.notna(p)]
                name = sep.join(parts) if parts else "عمود غير مسمى"

                if name in seen:
                    seen[name] += 1
                    name = f"{name}__{seen[name]}"
                else:
                    seen[name] = 1

                final_cols.append(name)

            df.columns = final_cols
            df.insert(0, "Sheet", sheet)

            self.sheets_dfs[sheet] = df.copy()

            sheet_index = i + 1
            sheet_label = f"sheet{sheet_index}"
            csv_name = f"{self.base_name}_{sheet_label}.csv"

            csv_path = self.output_dir / csv_name

            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"✓ Saved {csv_path} (rows={df.shape[0]}, cols={df.shape[1]})")

            frames.append(df)

        self.data_frame = pd.concat(frames, ignore_index=True)

        self.data_frame.to_csv(self.dataset_path_new, index=False, encoding="utf-8-sig")

        print("=" * 100)
        print(f"All sheets processed and saved to:\n{self.dataset_path_new}")
