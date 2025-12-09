import pandas as pd
class sorting_criteria: 
    def __init__(self, dataframepath, refrence_column, subject_columns,pass_threshold):
        self.dataset_path_new = dataframepath

        if isinstance(refrence_column, str):
            self.refrence_column = [refrence_column]
        else:
            self.refrence_column = list(refrence_column)
            
        self.SUBJECT_COLS = subject_columns
        self.PASS_THRESHOLD = pass_threshold

    def data_loading(self): 
        self.data_frame = pd.read_csv(self.dataset_path_new)
        self.data_frame.columns = self.data_frame.columns.str.strip()
        self.data_frame.columns = self.data_frame.columns.str.replace('\n', ' ', regex=True)

        print('=' * 100)
        print("Data maping in progres")
        print('=' * 100)

    def passed_rows(self):
        for column in self.SUBJECT_COLS:
            tag_col = f"{column} mapped"
            self.data_frame[tag_col] = (
                (self.data_frame[column] >= self.PASS_THRESHOLD)
                .fillna(False)     
                .astype("int8")
            )

        print("Data maping done")
        print('=' * 100)
        print(self.data_frame.info())
        print('=' * 100)

    def sorting_condition(self):
        data = self.data_frame.copy()

        loc_col = None
        for col in data.columns:
            if any(col.lower() == loc.lower() for loc in self.refrence_column):
                loc_col = col
                break
        if loc_col is None:
            raise ValueError("No location column found in dataset!")
        
        mapped_cols = [c for c in data.columns if "mapped" in c.lower()]
        if not mapped_cols:
            raise ValueError("No mapped pass/fail columns found! Run passed_rows() first.")

        data[mapped_cols] = data[mapped_cols].apply(pd.to_numeric, errors="coerce")

        per_subject_results = {}
        for subj, mapped_col in zip(self.SUBJECT_COLS, mapped_cols):
            grp = (
                data.groupby(loc_col, dropna=False)
                .agg(
                    n_students=(mapped_col, "count"),
                    pass_count=(mapped_col, "sum"),
                    mean_score=(subj, "mean")
                )
                .reset_index()
            )
            grp["pass_rate_pct"] = (grp["pass_count"] / grp["n_students"] * 100).round(2)
            grp["mean_score"] = grp["mean_score"].round(2)
            per_subject_results[subj] = grp

        overall_pass = (
            data.groupby(loc_col, dropna=False)[mapped_cols]
            .mean()
            .mul(100)
            .round(2)
            .reset_index()
        )
        overall_pass.columns = [loc_col] + [f"{col} %" for col in mapped_cols]

        self.per_subject_results = per_subject_results
        self.overall_pass = overall_pass

        print("=" * 100)
        print("Per-subject success rates by location:")

        for subj, df in per_subject_results.items():
            print(f"\nSubject: {subj}")
            print(df)

        print("=" * 100)
        print("Overall success rates by location:")
        print(overall_pass)
        print("=" * 100)

        return per_subject_results, overall_pass
