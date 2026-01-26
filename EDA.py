import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import warnings
import seaborn as sns

warnings.filterwarnings("ignore")


class EnhancedEduVisualizer:
    """
    Enhanced educational data visualizations with:
    - Value labels in legends
    - Optimized figure sizes
    - Column names as Y-axis labels
    - Bar plots with comparisons
    - Histograms for numerical data
    - ALL METHODS from CleanEduVisualizer (backward compatible)
    """

    def __init__(self, df, exclude_cols=None, force_numeric_cols=None, force_categorical_cols=None, auto_convert_numeric=True):
        self.df = df.copy()
        self.source_path = None
        self.auto_convert_numeric = auto_convert_numeric

        if exclude_cols is None:
            exclude_cols = []
        
        if force_numeric_cols is None:
            force_numeric_cols = []
        
        if force_categorical_cols is None:
            force_categorical_cols = []

        # Auto-detect ID-like columns
        auto_exclude = []
        for col in self.df.columns:
            col_lower = str(col).lower()
            if any(x in col_lower for x in ["id", "dob", "date of birth", "birth"]):
                auto_exclude.append(col)

        self.exclude_cols = list(set(exclude_cols + auto_exclude))
        self.force_numeric_cols = force_numeric_cols
        self.force_categorical_cols = force_categorical_cols
        
        # Auto-convert columns that look like numbers to numeric
        if self.auto_convert_numeric:
            self._convert_numeric_columns()
        
        self._categorize_columns()

        # Enhanced color palette - Professional and clean
        self.colors = {
            "primary": "#4BA3F1",
            "primary2": "#2D8BDC",
            "accent": "#7CC7FF",
            "glow": "#3BA0FF",
            "accent_red": "#FF6B7A",
            "accent_green": "#4ECDC4",
            "accent_orange": "#FFB84D",
            "accent_purple": "#A78BFA",
            "dark_text": "#2C3E50",
            "light_text": "#7F8C8D",
        }

        # Set seaborn style for clean, professional plots
        sns.set_style("whitegrid", {
            'axes.edgecolor': '#E0E0E0',
            'grid.color': '#F0F0F0',
            'grid.linestyle': '--',
        })
        sns.set_context("notebook", font_scale=1.1)

        print("📊 Dataset Overview:")
        print(f"   Total columns: {len(self.df.columns)}")
        print(f"   Excluded columns: {self.exclude_cols}")
        print(f"   Numeric columns: {self.numeric_cols}")
        print(f"   Categorical columns: {self.categorical_cols}")

    def _convert_numeric_columns(self):
        """Attempt to convert string columns that look like numbers to numeric"""
        for col in self.df.columns:
            if col not in self.exclude_cols:
                # Check if column is currently object/string type
                if self.df[col].dtype == 'object':
                    # Try to convert to numeric, keeping errors as NaN
                    converted = pd.to_numeric(self.df[col], errors='coerce')
                    # If successful conversion (not all NaN), update the column
                    if not converted.isna().all():
                        # Check if the conversion was successful for most values
                        success_ratio = 1 - (converted.isna().sum() / len(converted))
                        if success_ratio > 0.8:  # At least 80% of values successfully converted
                            self.df[col] = converted
                            print(f"   🔄 Converted '{col}' to numeric (success: {success_ratio:.1%})")

    def set_source_path(self, source_path: str | None):
        from pathlib import Path
        self.source_path = Path(source_path) if source_path else None

    def _default_save_dir(self):
        if self.source_path:
            out = self.source_path.parent / f"{self.source_path.stem}_plots"
            out.mkdir(exist_ok=True)
            return str(out)
        return None

    def _categorize_columns(self):
        all_cols = [col for col in self.df.columns if col not in self.exclude_cols]
        self.numeric_cols = []
        self.categorical_cols = []

        for col in all_cols:
            # Check if column is forced to be numeric
            if col in self.force_numeric_cols:
                self.numeric_cols.append(col)
            # Check if column is forced to be categorical
            elif col in self.force_categorical_cols:
                self.categorical_cols.append(col)
            # Check if column is numeric data type (int, float, etc.)
            elif pd.api.types.is_numeric_dtype(self.df[col]):
                # Pure data type logic - if it's numeric type, treat as numeric
                self.numeric_cols.append(col)
            else:
                # Non-numeric data type - treat as categorical
                self.categorical_cols.append(col)
        
        # Debug: Print data types for all columns
        print("🔍 Column Data Types:")
        for col in all_cols:
            dtype = self.df[col].dtype
            classification = "NUMERIC" if col in self.numeric_cols else "CATEGORICAL"
            print(f"   {col}: {dtype} -> {classification}")

    def _clean_label(self, text):
        if text is None:
            return ""
        text = str(text).strip().replace("_", " ").replace("-", " ")
        return " ".join(word.capitalize() for word in text.split())

    def _create_palette(self, n):
        """Create gradient palette using darker primary colors"""
        if n <= 1:
            return [self.colors["primary"]]
        # Use darker colors for better contrast
        cmap = LinearSegmentedColormap.from_list(
            "edu_blues", [self.colors["primary"], self.colors["primary2"], "#1a5f9e", "#0d3d66"], N=n
        )
        return [cmap(i / (n - 1)) for i in range(n)]

    def _safe_save(self, fig, save_path: str | None):
        if not save_path:
            return
        fig.patch.set_facecolor("white")
        for ax in fig.axes:
            try:
                ax.set_facecolor("white")
            except:
                pass
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white", transparent=False)

    # ==================== CATEGORICAL PLOTS ====================
    
    def plot_categorical_bar(self, column, save_path=None, top_n=15):
        """Bar plot for categorical data - horizontal bars"""
        if column not in self.df.columns:
            print(f"❌ Column '{column}' not found")
            return

        # Get counts
        counts = self.df[column].value_counts().sort_values(ascending=True)
        
        # Handle large number of categories
        if len(counts) > top_n:
            print(f"   ℹ️  Showing top {top_n} categories out of {len(counts)} total")
            # Keep top N and combine rest into "Others"
            top_counts = counts.tail(top_n - 1)
            others_count = counts.iloc[:-top_n+1].sum()
            
            # Add "Others" category
            top_counts = pd.concat([pd.Series([others_count], index=['Others']), top_counts])
            counts = top_counts

        fig, ax = plt.subplots(figsize=(12, max(6, len(counts) * 0.5)))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        colors = self._create_palette(len(counts))
        
        # Create horizontal bars
        bars = ax.barh(range(len(counts)), counts.values, color=colors, 
                      edgecolor="white", linewidth=2.5, alpha=0.85)

        # Add value labels at the end of bars
        for i, (bar, val) in enumerate(zip(bars, counts.values)):
            width = bar.get_width()
            ax.text(width, i, f' {int(val)}', 
                   ha='left', va='center', 
                   fontsize=11, fontweight='600', color=self.colors['dark_text'])

        ax.set_yticks(range(len(counts)))
        ax.set_yticklabels(counts.index, fontsize=11, fontweight='500')
        ax.set_xlabel('Count', fontsize=14, fontweight='600', color=self.colors['dark_text'])
        ax.set_ylabel(self._clean_label(column), fontsize=13, fontweight='500', color=self.colors['light_text'])
        ax.set_title(f'Distribution of {self._clean_label(column)}', 
                    fontsize=18, fontweight='bold', color=self.colors['dark_text'], pad=20)
        
        # Clean spines
        sns.despine(ax=ax, left=False, bottom=True)
        ax.tick_params(labelsize=10, colors=self.colors['light_text'])
        ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.8)

        plt.tight_layout()
        self._safe_save(fig, save_path)
        plt.show()

    def plot_categorical_pie(self, column, max_unique=20, save_path=None):
        """Pie chart for categorical data with legend"""
        if column not in self.df.columns:
            print(f"❌ Column '{column}' not found")
            return

        # Check if column has too many unique values for a pie chart
        unique_count = self.df[column].nunique()
        if unique_count > max_unique:
            print(f"⚠️  Column '{column}' has {unique_count} unique values, exceeding the limit of {max_unique} for pie chart")
            return

        print(f"✅ Creating pie chart for '{column}' with {unique_count} unique values")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # Get counts
        counts = self.df[column].value_counts().sort_values(ascending=False)

        colors = self._create_palette(len(counts))
        
        # Create pie chart without labels on the pie itself
        wedges, texts, autotexts = ax.pie(counts.values, autopct='%1.1f%%',
                                        colors=colors, startangle=90,
                                        wedgeprops={'edgecolor': 'white', 'linewidth': 2.5},
                                        textprops={'fontsize': 10, 'fontweight': '600'})
        
        # Style percentage text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        # Create legend with counts
        legend_labels = [f'{cat}: {int(val)} ({val/counts.sum()*100:.1f}%)' 
                        for cat, val in zip(counts.index, counts.values)]
        ax.legend(wedges, legend_labels, title=self._clean_label(column),
                 loc='center left', bbox_to_anchor=(1, 0, 0.5, 1),
                 fontsize=10, frameon=True, fancybox=True)
        
        ax.set_title(f'Distribution of {self._clean_label(column)}', 
                    fontsize=18, fontweight='bold', color=self.colors['dark_text'], pad=20)
        
        plt.tight_layout()
        self._safe_save(fig, save_path)
        plt.show()

    # ==================== NUMERICAL PLOTS ====================
    
    def plot_numeric_histogram(self, column, bins=30, save_path=None):
        """Enhanced histogram for numerical data with KDE"""
        if column not in self.df.columns:
            print(f"❌ Column '{column}' not found")
            return

        fig, ax = plt.subplots(figsize=(14, 8))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        data = self.df[column].dropna()
        
        # Create histogram manually to apply gradient
        n, bins_edges, patches = ax.hist(data, bins=bins, edgecolor='white', 
                                        linewidth=1.5, alpha=0.8)

        # Apply gradient colors to bars
        colors = self._create_palette(len(patches))
        for patch, color in zip(patches, colors):
            patch.set_facecolor(color)

        # Add KDE line
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data, bw_method='scott')
            x_range = np.linspace(data.min(), data.max(), 500)
            kde_values = kde(x_range)
            # Scale KDE to match histogram height
            kde_scaled = kde_values * len(data) * (bins_edges[1] - bins_edges[0])
            ax.plot(x_range, kde_scaled, color=self.colors['glow'], linewidth=3.5, 
                   alpha=0.9, zorder=10)
        except:
            pass

        # Calculate statistics
        mean_val = data.mean()
        median_val = data.median()
        std_val = data.std()
        
        # Add vertical lines for mean and median
        mean_line = ax.axvline(mean_val, color=self.colors['accent_red'], linestyle='--', 
                  linewidth=2.5, alpha=0.8)
        median_line = ax.axvline(median_val, color=self.colors['accent_green'], linestyle='--', 
                  linewidth=2.5, alpha=0.8)

        # Add statistics box
        stats_text = (f'n = {len(data):,}\n'
                     f'μ = {mean_val:.2f}\n'
                     f'σ = {std_val:.2f}\n'
                     f'min = {data.min():.2f}\n'
                     f'max = {data.max():.2f}')
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               ha='left', va='top', fontsize=11, color=self.colors['dark_text'],
               fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.8', facecolor='white', 
                        edgecolor=self.colors['primary'], alpha=0.9, linewidth=2))

        ax.set_ylabel('Frequency', fontsize=14, fontweight='600', color=self.colors['dark_text'])
        ax.set_xlabel(self._clean_label(column), fontsize=13, fontweight='500', color=self.colors['light_text'])
        ax.set_title(f'Distribution of {self._clean_label(column)}', 
                    fontsize=18, fontweight='bold', color=self.colors['dark_text'], pad=20)
        
        # Create custom legend with better positioning
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=self.colors['accent_red'], linestyle='--', linewidth=2.5, label=f'Mean: {mean_val:.2f}'),
            Line2D([0], [0], color=self.colors['accent_green'], linestyle='--', linewidth=2.5, label=f'Median: {median_val:.2f}')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=11, frameon=True, 
                 fancybox=True, framealpha=0.95)
        
        sns.despine(ax=ax, left=True)
        ax.tick_params(labelsize=10, colors=self.colors['light_text'])
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)

        plt.tight_layout()
        self._safe_save(fig, save_path)
        plt.show()

    # ==================== COMPARISON PLOTS ====================
    
    def plot_comparison_by_category(self, numeric_cols, category_col, save_path=None):
        """Compare multiple numeric columns grouped by a category"""
        if category_col not in self.df.columns:
            print(f"❌ Category column '{category_col}' not found")
            return
        
        valid_cols = [col for col in numeric_cols if col in self.df.columns]
        if not valid_cols:
            print("❌ No valid numeric columns found")
            return

        fig, ax = plt.subplots(figsize=(16, 8))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # Get categories
        categories = sorted(self.df[category_col].dropna().unique())
        x = np.arange(len(categories))
        width = 0.8 / len(valid_cols)

        # Color palette for subjects
        subject_colors = [
            self.colors['primary'], self.colors['accent'], 
            self.colors['accent_orange'], self.colors['accent_purple'],
            self.colors['accent_green'], self.colors['accent_red']
        ]

        # Plot each numeric column
        for i, col in enumerate(valid_cols):
            means = []
            for cat in categories:
                cat_data = self.df[self.df[category_col] == cat][col].dropna()
                means.append(cat_data.mean() if len(cat_data) > 0 else 0)
            
            offset = (i - len(valid_cols)/2 + 0.5) * width
            bars = ax.bar(x + offset, means, width, 
                         label=self._clean_label(col),
                         color=subject_colors[i % len(subject_colors)], 
                         edgecolor='white', linewidth=2, alpha=0.85)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, height,
                           f'{height:.1f}', ha='center', va='bottom', 
                           fontsize=9, fontweight='600', color=self.colors['dark_text'])

        ax.set_xticks(x)
        ax.set_xticklabels([str(cat) for cat in categories], fontsize=11, fontweight='500')
        ax.set_ylabel('Average Score', fontsize=14, fontweight='600', color=self.colors['dark_text'])
        ax.set_xlabel(self._clean_label(category_col), fontsize=13, fontweight='500', color=self.colors['light_text'])
        ax.set_title(f'Comparison of Scores by {self._clean_label(category_col)}', 
                    fontsize=18, fontweight='bold', color=self.colors['dark_text'], pad=20)
        
        ax.legend(loc='upper right', fontsize=11, frameon=True, fancybox=True, ncol=2)
        sns.despine(ax=ax, left=True)
        ax.tick_params(labelsize=10, colors=self.colors['light_text'])
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)

        plt.tight_layout()
        self._safe_save(fig, save_path)
        plt.show()

    def plot_double_comparison(self, col1, col2, category_col, save_path=None):
        """Compare two numeric columns side by side"""
        if category_col not in self.df.columns or col1 not in self.df.columns or col2 not in self.df.columns:
            print(f"❌ Required columns not found")
            return

        fig, ax = plt.subplots(figsize=(14, 8))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        categories = sorted(self.df[category_col].dropna().unique())
        x = np.arange(len(categories))
        width = 0.35

        # Calculate means
        means1 = [self.df[self.df[category_col] == cat][col1].mean() for cat in categories]
        means2 = [self.df[self.df[category_col] == cat][col2].mean() for cat in categories]

        # Create bars
        bars1 = ax.bar(x - width/2, means1, width, 
                      label=self._clean_label(col1),
                      color=self.colors['primary'], edgecolor='white', linewidth=2, alpha=0.85)
        bars2 = ax.bar(x + width/2, means2, width, 
                      label=self._clean_label(col2),
                      color=self.colors['accent_green'], edgecolor='white', linewidth=2, alpha=0.85)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height,
                       f'{height:.1f}', ha='center', va='bottom', 
                       fontsize=10, fontweight='600', color=self.colors['dark_text'])

        ax.set_xticks(x)
        ax.set_xticklabels([str(cat) for cat in categories], fontsize=11, fontweight='500')
        ax.set_ylabel('Average Score', fontsize=14, fontweight='600', color=self.colors['dark_text'])
        ax.set_xlabel(self._clean_label(category_col), fontsize=13, fontweight='500', color=self.colors['light_text'])
        ax.set_title(f'{self._clean_label(col1)} vs {self._clean_label(col2)} by {self._clean_label(category_col)}', 
                    fontsize=18, fontweight='bold', color=self.colors['dark_text'], pad=20)
        
        ax.legend(loc='upper right', fontsize=12, frameon=True, fancybox=True)
        sns.despine(ax=ax, left=True)
        ax.tick_params(labelsize=10, colors=self.colors['light_text'])
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)

        plt.tight_layout()
        self._safe_save(fig, save_path)
        plt.show()

    # ==================== GENERATE ALL PLOTS ====================
    
    def generate_all_plots(self, save_dir=None):
        """Generate all visualizations for the dataset"""
        print("\n" + "="*70)
        print("🎨 GENERATING ALL VISUALIZATIONS")
        print("="*70)
        
        if save_dir is None:
            save_dir = self._default_save_dir()
        
        plot_count = 0
        
        # 1. All numeric columns (histograms)
        print(f"\n📈 Numeric Histograms ({len(self.numeric_cols)} columns):")
        for i, col in enumerate(self.numeric_cols, 1):
            print(f"   [{i}/{len(self.numeric_cols)}] {col}")
            save_path = f"{save_dir}/num_{i:02d}_{col.replace(' ', '_')}.png" if save_dir else None
            try:
                self.plot_numeric_histogram(col, save_path=save_path)
                plot_count += 1
            except Exception as e:
                print(f"      ⚠️ Error: {e}")
        
        # 2. All categorical columns (bar charts and pie charts)
        print(f"\n📊 Categorical Plots ({len(self.categorical_cols)} columns):")
        for i, col in enumerate(self.categorical_cols, 1):
            print(f"   [{i}/{len(self.categorical_cols)}] {col}")
            # Bar chart
            bar_path = f"{save_dir}/cat_bar_{i:02d}_{col.replace(' ', '_')}.png" if save_dir else None
            try:
                self.plot_categorical_bar(col, save_path=bar_path, top_n=15)
                plot_count += 1
            except Exception as e:
                print(f"      ⚠️ Error bar: {e}")
            
            # Pie chart (with condition)
            pie_path = f"{save_dir}/cat_pie_{i:02d}_{col.replace(' ', '_')}.png" if save_dir else None
            try:
                self.plot_categorical_pie(col, max_unique=20, save_path=pie_path)
                plot_count += 1
            except Exception as e:
                print(f"      ⚠️ Error pie: {e}")
        
        # 3. Comparison plots (if we have appropriate columns)
        if self.categorical_cols and len(self.numeric_cols) >= 2:
            print(f"\n🔄 Comparison Plots:")
            category_col = self.categorical_cols[0]  # Use first categorical column
            
            # Multi-subject comparison
            print(f"   Generating multi-subject comparison by {category_col}")
            save_path = f"{save_dir}/comparison_all_by_{category_col}.png" if save_dir else None
            try:
                self.plot_comparison_by_category(self.numeric_cols[:6], category_col, save_path)
                plot_count += 1
            except Exception as e:
                print(f"      ⚠️ Error: {e}")
            
            # Pairwise comparisons
            for i in range(0, len(self.numeric_cols)-1, 2):
                col1, col2 = self.numeric_cols[i], self.numeric_cols[i+1]
                print(f"   Comparing {col1} vs {col2}")
                save_path = f"{save_dir}/comparison_{col1}_vs_{col2}.png" if save_dir else None
                try:
                    self.plot_double_comparison(col1, col2, category_col, save_path)
                    plot_count += 1
                except Exception as e:
                    print(f"      ⚠️ Error: {e}")
        
        print("\n" + "="*70)
        print(f"✅ COMPLETE! Generated {plot_count} visualizations")
        if save_dir:
            print(f"📁 Saved to: {save_dir}/")
        print("="*70)


# ==================== USAGE EXAMPLE ====================
if __name__ == "__main__":
    # Load your data
    df = pd.read_csv("your_data.csv")  # Replace with your file
    
    # Create visualizer
    viz = EnhancedEduVisualizer(df, exclude_cols=['student_id', 'ID'])
    
    # Option 1: Generate all plots automatically
    viz.generate_all_plots(save_dir="output_plots")
    
    # Option 2: Generate specific plots
    # viz.plot_categorical_bar('grade', save_path='grade_dist.png')
    # viz.plot_numeric_histogram('math_total', save_path='math_hist.png')
    # viz.plot_comparison_by_category(['arabic_total', 'english_total', 'math_total'], 
    #                                 'grade', save_path='comparison.png')
    # viz.plot_double_comparison('arabic_total', 'english_total', 'grade')