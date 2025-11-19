from dataset_loading import Data_set
from data_preperation import Data_analysis
from EDA import data_visualisation
from sorting_conditons import sorting_criteria

from config import (
    Old_file, New_file, sheets, SUBJECT_FINAL_COLS, 
    THRESHOLDS, REFREBCE_COLUMN, TLS_NAME_MAP, common, 
    plot_defaults, GROUPING_COLUMN, TOTAL_MARK, 
    multiple_bar_charts_xlabel, multiple_bar_charts_ylabel,
    PIE_MAP, PIE_ORDER, 
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.pause = lambda *a, **k: None

import sys
import datetime
import io
from pathlib import Path
import pandas as pd
import traceback
import numpy as np


class Logger:
    def __init__(self, log_file: str = "Results.txt"):
        self.log_file = Path(log_file)
        self.log_stream = None
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        
    def __enter__(self):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_stream = open(self.log_file, "w", encoding="utf-8")
        
        sys.stdout = self._Tee(self._original_stdout, self.log_stream)
        sys.stderr = self._Tee(self._original_stderr, self.log_stream)
        
        print(f"\n{'='*100}")
        print(f"Logging started at {timestamp}")
        print(f"{'='*100}\n")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            print(f"\n{'='*100}")
            print(f"ERROR: {exc_type.__name__}: {exc_val}")
            traceback.print_tb(exc_tb)
            print(f"{'='*100}\n")
        
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        
        if self.log_stream:
            self.log_stream.close()
    
    class _Tee(io.TextIOBase):
        def __init__(self, *streams):
            self.streams = streams
            
        def write(self, s):
            for stream in self.streams:
                stream.write(s)
                stream.flush()
            return len(s)
            
        def flush(self):
            for stream in self.streams:
                stream.flush()


class DataPipeline:
    
    def __init__(self, excel_file: str, csv_file: str, sheets_config: list):
        self.excel_file = excel_file
        self.csv_file = csv_file
        self.sheets_config = sheets_config
        self.common_plot_params = common
        self.plot_defaults = plot_defaults
        
    def load_dataset(self):
        print(f"\n{'='*100}")
        print("Dataset Loading")
        print(f"{'='*100}\n")
        
        try:
            dataset = Data_set(self.excel_file, self.csv_file)
            dataset.load_excel_with_multilevel_header()
            print("✓ Dataset loaded successfully")
        except Exception as e:
            print(f"⚠ Warning: Could not load dataset: {e}")
            print("Continuing with existing CSV files...")
    
    def clean_data(self, csv_path: str) -> str:
        print(f"\n{'='*100}")
        print("Dataset Cleaning ")
        print(f"{'='*100}\n")

        p = Path(csv_path)
        if not p.exists():

            candidate = p.with_suffix('') / p.name
            if candidate.exists():
                p = candidate
            else:
                raise FileNotFoundError(
                    f"CSV not found at '{csv_path}' or '{candidate}'. "
                    "Update config.sheets or keep this resolver."
                )

        analyzer = Data_analysis(str(p))
        analyzer.data_loading()
        analyzer.column_rename()
        analyzer.data_investigation()
        analyzer.data_cleaning()

        cleaned_path = p.parent / f"{p.stem}_cleaned.csv"
        print(f"✓ Cleaned file saved: {cleaned_path}")
        return str(cleaned_path)
    
    def process_threshold(self, csv_path: str, threshold: int) -> pd.DataFrame:
        sorter = sorting_criteria(
            dataframepath=csv_path,
            refrence_column=REFREBCE_COLUMN,
            subject_columns=SUBJECT_FINAL_COLS,
            pass_threshold=threshold,
        )

        sorter.data_loading()
        self._validate_columns(sorter.data_frame, csv_path)

        if GROUPING_COLUMN not in sorter.data_frame.columns and REFREBCE_COLUMN in sorter.data_frame.columns:
            sorter.data_frame[GROUPING_COLUMN] = (
                sorter.data_frame[REFREBCE_COLUMN]
                .map(TLS_NAME_MAP)
                .fillna(sorter.data_frame[REFREBCE_COLUMN])
            )

        sorter.passed_rows()

        df = sorter.data_frame.copy()

        mapped_cols = [f"{s} mapped" for s in SUBJECT_FINAL_COLS]
        for col in mapped_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int8")

        df[GROUPING_COLUMN] = df[REFREBCE_COLUMN].map(TLS_NAME_MAP).fillna(df[REFREBCE_COLUMN])
        df["Location"] = df[GROUPING_COLUMN]

        if all(c in df.columns for c in mapped_cols):
            df["All Subjects Passed"] = df[mapped_cols].eq(1).all(axis=1).astype("int8")
        else:
            missing = [c for c in mapped_cols if c not in df.columns]
            print(f"⚠ Missing mapped columns for 'All Subjects Passed': {missing}")


        sorter.data_frame = df

        self._print_success_tables(sorter, threshold)
        return df

    
    def _validate_columns(self, df: pd.DataFrame, csv_path: str):
        required_cols = [REFREBCE_COLUMN] + SUBJECT_FINAL_COLS
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"\n{'='*100}")
            print(f"ERROR: Missing required columns in {csv_path}")
            print(f"Missing: {missing_cols}")
            print(f"Available: {df.columns.tolist()}")
            print(f"{'='*100}\n")
            raise KeyError(f"Missing columns: {missing_cols}")
    
    def _print_success_tables(self, sorter: sorting_criteria, threshold: int):
        per_subject, overall = sorter.sorting_condition()
        
        print(f"\n{'='*100}")
        print(f"SUCCESS RATES @ Threshold ≥ {threshold} / {TOTAL_MARK}")
        print(f"{'='*100}\n")
        
        for subject, table in per_subject.items():
            print(f"Subject: {subject}")
            print(table)
            print()
        
        print("Overall Success by Location:")
        print(overall)
        print(f"{'='*100}\n")
    
    def plot_score_distributions(self, csv_path: str, df: pd.DataFrame, grade_label: str):
        print(f"\n Plotting Score Distributions")
        
        vis = data_visualisation(csv_path)
        vis.data_loading()
        vis.data_frame = df.copy()
        vis.data_frame.columns = vis.data_frame.columns.str.strip()
        
        for col in SUBJECT_FINAL_COLS:
            if col not in df.columns:
                print(f"⚠ Warning: Column '{col}' not found. Skipping.")
                continue
            
            params = {
                **self.common_plot_params,
                **self.plot_defaults["histogram"],
                "column": col,
                "chart_name": f"{col} Distribution — {grade_label}",
                "xlabel": col,
                "ylabel": "Frequency",
                "label": f"hist_{grade_label}_{col.replace(' ', '_')}",
            }
            
            vis.histogram(**params)
            print(f"✓ Created histogram for {col}")
    
    def plot_success_rates(self, csv_path: str, df: pd.DataFrame, 
                          threshold: int, grade_label: str):
        print(f"\n Plotting Success Rates")
        
        vis = data_visualisation(dataset_path=csv_path, df=df)
        vis.data_loading()  
        vis.data_frame.columns = vis.data_frame.columns.str.strip()

        
        required_cols = [GROUPING_COLUMN] + [f"{s} mapped" for s in SUBJECT_FINAL_COLS]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"⚠ Warning: Cannot plot success rates. Missing: {missing_cols}")
            return
        
        vis.multiple_bar_charts(
            chart_name=f"Success Rate (≥ {threshold}/ {TOTAL_MARK}) by Location and Subject — {grade_label}",
            group_col=GROUPING_COLUMN,
            score_cols=[f"{s} mapped" for s in SUBJECT_FINAL_COLS],
            agg_func="mean",
            xlabel=multiple_bar_charts_xlabel,
            ylabel=multiple_bar_charts_ylabel,
            ylim=(0, 1.5),
            annotate=True,
            annotate_format="{:.0%}",
            **self.common_plot_params,
        )
        
        print(f"✓ Created success rate chart for threshold {threshold}")

    
    def plot_pie_charts(self, csv_path: str, df: pd.DataFrame, grade_label: str):
        print(f"\n Plotting Pie Charts")

        vis = data_visualisation(csv_path)
        vis.data_loading()
        vis.data_frame = df.copy()

        for col, title in PIE_MAP.items():
            if col not in df.columns:
                print(f"⚠ Skipping pie for '{col}': column not found.")
                continue

            if col in PIE_ORDER:
                ordered = pd.Categorical(df[col], categories=PIE_ORDER[col], ordered=True)
                vis.data_frame[col] = ordered

            n_unique = df[col].nunique(dropna=False)
            max_slices = self.plot_defaults.get("pie_chart", {}).get("max_slices", 20)
            if n_unique == 0 or n_unique > max_slices:
                print(f"⚠ Skipping pie for '{col}': unsuitable number of categories ({n_unique}).")
                continue

            vis.pie_chart(
                chart_name=f"{title} — {grade_label}",
                column=col,
                **self.common_plot_params,
                **self.plot_defaults.get("pie_chart", {})
            )
            print(f"✓ Created pie chart for {col}")

    def plot_box_plots(self, csv_path: str, df: pd.DataFrame, grade_label: str):
        print(f"\n Plotting Box Plots")
        
        vis = data_visualisation(csv_path)
        vis.data_loading()
        vis.data_frame = df.copy()
        
        for col in SUBJECT_FINAL_COLS:
            if col not in df.columns:
                continue
            
            vis.box_plot(
                chart_name=f"{col} Distribution by Location — {grade_label}",
                column=col,
                group_by=GROUPING_COLUMN,
                xlabel=GROUPING_COLUMN,
                ylabel=f"{col} Score",
                **self.common_plot_params
            )
            print(f"✓ Created box plot for {col}")
    
    def plot_correlation_heatmap(self, csv_path: str, df: pd.DataFrame, grade_label: str):
        
        print(f"\n Plotting Correlation Heatmap")
        
        vis = data_visualisation(csv_path)
        vis.data_loading()
        vis.data_frame = df.copy()
        
        numeric_cols = [col for col in SUBJECT_FINAL_COLS if col in df.columns]
        
        if len(numeric_cols) >= 2:
            vis.correlation_heatmap(
                columns=numeric_cols,
                chart_name=f"Subject Score Correlations — {grade_label}",
                **self.common_plot_params
            )
            print(f"✓ Created correlation heatmap")
        else:
            print(f"⚠ Warning: Not enough numeric columns for correlation")
    
    def plot_value_counts_charts(self, csv_path: str, df: pd.DataFrame, grade_label: str):
        """Bar charts for categorical distributions (1 variable)"""
        print(f"\n Plotting Value Count Charts")

        vis = data_visualisation(csv_path)
        vis.data_loading()
        vis.data_frame = df.copy()

        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        categorical_cols = [col for col in categorical_cols if "mapped" not in col.lower()]

        for col in categorical_cols[:5]:
            n_unique = df[col].nunique(dropna=False)
            if n_unique > 20:
                print(f"⚠ Skipping {col}: too many categories ({n_unique})")
                continue

            vis.bar_chart(
                chart_name=f"Distribution of {col} — {grade_label}",
                column=col,
                sort_by="y",
                ascending=False,
                annotate=True,
                annotate_format="{:,.0f}",
                **self.common_plot_params
            )
            print(f"✓ Created bar chart for {col}")

    def export_summary_reports(self, csv_path: str, df: pd.DataFrame, grade_label: str):
        print(f"\n Exporting Summary Reports")
        
        vis = data_visualisation(csv_path)
        vis.data_loading()
        vis.data_frame = df.copy()
        
        output_file = Path(csv_path).parent / f"summary_report_{grade_label}.txt"
        vis.export_summary_report(output_file=output_file, include_plots=False)
        print(f"✓ Summary report exported: {output_file}")
    
    def plot_univariate_scatter_distributions(self, csv_path: str, df: pd.DataFrame, grade_label: str):
        print(f"\n Plotting Univariate Scatter Distributions")

        vis = data_visualisation(csv_path)
        vis.data_loading()
        vis.data_frame = df.copy()
        vis.data_frame.columns = vis.data_frame.columns.str.strip()

        for col in SUBJECT_FINAL_COLS:
            if col not in df.columns:
                print(f"⚠ Warning: Column '{col}' not found. Skipping.")
                continue

            vis.univariate_scatter(
                chart_name=f"{col} Distribution — {grade_label}",
                column=col,      
                point_size=40,
                alpha=0.9,
                jitter=0.06,
                orientation="v",
                **self.common_plot_params
            )
            print(f"✓ Created univariate scatter for {col}")

    def plot_all_visualizations(self, csv_path: str, df: pd.DataFrame, grade_label: str):
        print(f"\n{'='*100}")
        print(f"GENERATING ALL VISUALIZATIONS FOR {grade_label}")
        print(f"{'='*100}\n")
        
        try:
            self.plot_score_distributions(csv_path, df, grade_label) 
            self.plot_univariate_scatter_distributions(csv_path, df, grade_label)
            self.plot_box_plots(csv_path, df, grade_label)
            self.plot_pie_charts(csv_path, df, grade_label)
            self.plot_correlation_heatmap(csv_path, df, grade_label)
            self.export_summary_reports(csv_path, df, grade_label)
            
            print(f"\n✓ All visualizations completed for {grade_label}")
            
        except Exception as e:
            print(f"⚠ Warning: Error in visualization generation: {e}")
            traceback.print_exc()
    
    def process_grade(self, csv_path: str, grade_label: str):
        print(f"\n{'='*100}")
        print(f"PROCESSING: {grade_label} ({csv_path})")
        print(f"{'='*100}\n")

        p = Path(csv_path)
        if not p.exists():
            candidate = p.with_suffix('') / p.name
            if candidate.exists():
                csv_path = str(candidate)
            else:
                print(f"⚠ Skipping {grade_label}: could not find '{csv_path}' or '{candidate}'")
                return

        try:
            cleaned_path = self.clean_data(csv_path)

            plotted_distributions = False
            for threshold in THRESHOLDS:
                print(f"\n{'─'*100}")
                print(f"Threshold: {threshold}/20")
                print(f"{'─'*100}")

                df = self.process_threshold(cleaned_path, threshold)

                if not plotted_distributions:
                    self.plot_all_visualizations(cleaned_path, df, grade_label)
                    plotted_distributions = True
                    
                self.plot_success_rates(cleaned_path, df, threshold, grade_label)

            print(f"\n✓ Successfully completed {grade_label}")

        except Exception as e:
            print(f"\n{'='*100}")
            print(f"ERROR processing {grade_label}: {e}")
            print(f"{'='*100}\n")
            traceback.print_exc()
            print(f"\nSkipping {grade_label} and continuing...\n")
    
    def run(self):
        self.load_dataset()
        
        for csv_path, grade_label in self.sheets_config:
            self.process_grade(csv_path, grade_label)
        
        print(f"\n{'='*100}")
        print("PIPELINE COMPLETE!")
        print(f"{'='*100}\n")


def main():
    with Logger("Results.txt"):
        pipeline = DataPipeline(
            excel_file=Old_file,
            csv_file=New_file,
            sheets_config=sheets
        )
        pipeline.run()


if __name__ == "__main__":
    main()


