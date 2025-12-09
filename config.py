import re
import unicodedata

Old_file = "7 gr.xlsx"
New_file = "Seventh Grade.csv"

# Sheets name
sheets = [
    ("Dataframe_first-sheet.csv",  "Sheet1")
]

# Columns to drop
columns_to_drop = ["Student Full Name", "Date of Birth", "Sheet", "English Total Score as (%)", "Arabic Total Score as (%)", "Math Total Score as (%)"]

# Bar chart entries 
SUBJECT_FINAL_COLS = ["English Total Score", "Arabic Total Score", "Math Total Score"]

# Pie chart column and title
PIE_MAP = {
    "TLS Name": "Student Distribution by Location",
    "Gender": "Gender Distribution",
    "Math Level": "Math Level (Levels 1–4)",
    "Arabic Reading Category": "Arabic Reading Category",
    "English Reading & Vocabulary Category": "English Reading & Vocabulary Category",
    "TLS Name": "Student Distribution by Location", 
    }
# Pie chart Categorical lables
PIE_ORDER = {
    "Math Level": ["Level 1", "Level 2", "Level 3", "Level 4"],
    "Arabic Reading Category": ["Level 1", "Level 2", "Level 3", "Level 4"],
    "English Reading & Vocabulary Category": ["Level 1", "Level 2", "Level 3", "Level 4"],
    "Gender": ["Male", "Female"],
}

# Threshold labels 
TOTAL_MARK = 20
THRESHOLDS = [10, 15]

# Refrence threshold columns
REFREBCE_COLUMN = "TLS Name"
GROUPING_COLUMN = "TLS Name"

# Two bar chart columns: 
multiple_bar_charts_xlabel = "Location (TLS)"
multiple_bar_charts_ylabel = "Success Rate (%)"

# Map to clean, grouped location names
TLS_NAME_MAP = {
    "Sonna' Al Ebtisama": "Sonna' Al Ebtisama",
    "Zahrat Al-Madina": "Zahrat Al-Madina",
    "Al-Buraq Center": "Al-Buraq Center",
}

# Used fonts and plots settings
COMMON_FONT = {
    "family": "Times New Roman",
    "title_size": 24,
    "label_size": 18,
    "tick_size": 14,
    "value_label_size": 12,
}

common = {
    "size": (14, 8),
    "font": COMMON_FONT,
}

plot_defaults = {
    "histogram": {
        "bins": 20,
        "kde": True,
        "show_mean_line": True,
        "show_stats_box": True,
        "annotate": False,
    },
    
    "multiple_bar_charts": {},
    
    "bar_chart": {
        "annotate": True,
        "annotate_format": "{:,.0f}",
    },
    
    "pie_chart": {
        "show_legend": True,
        "explode_top_n": 0,
        "max_slices": 20,
    },
}

# Column rename
COLUMN_NAME_MAP = {
    "Sheet": "Sheet",
    "Full Name of the Student": "Student Full Name",
    "Gender": "Gender",
    "Date of Birth": "Date of Birth",
    "TLS": "TLS Name",
    "Did the student transfer from another TLS": "Transferred from Another TLS",
    "Approximate Number of Attendance Days": "Attendance Days",
    "Any Disability/Injury/Chronic Illness": "Disability or Chronic Illness",
    "Has the student lost a first-degree relative (mother, father, sibling, other)": "First-Degree Relative Loss",
    "Has the student lost a first-degree relative (mother, father, sibling, other": "First-Degree Relative Loss",
    
    "math": "Math Total Score",
    "math 7 th": "Math Level",
    "math 100%": "Math Total Score as (%)",

    "arabic Reading and Comprehension": "Arabic Reading Score",
    "Reading and Comprehension Classification": "Arabic Reading Category",
    "arabic Language Exercises": "Arabic Exercises Score",
    "arabic Language Exercises Classification": "Arabic Level",
    "arabicLanguage Exercises Classification": "Arabic Level",
    "Total Score in Arabic Language": "Arabic Total Score",
    "arabic 100%": "Arabic Total Score as (%)",
    
    "Reading comprehesion & vocabulary": "English Reading & Vocabulary Score",
    "English Reading comprehesion & vocabulary": "English Reading & Vocabulary Category",
    "English Grammar": "English Grammar Score",
    "Language Classification": "English Level",
    "Total Score in English Language": "English Total Score",
    "english 100%": "English Total Score as (%)",
    
    "AVERAGE_NUMERIC": "Average Score",    
    "AVERAGE_CATEGORY": "Average Category",  
}

# Entry rename 
COLUMN_VALUE_MAP = {
    # Demographics
    "Gender": {
        'male': 'Male',
        'Female': 'Female',
    },
    
    "TLS Name": {
        "مركز صناع الامل": "Sonna' Al Ebtisama",
        "زهرات المدينة": "Zahrat Al-Madina",
        "مركز البراق التعليمي": "Al-Buraq Center",
    },
    
    "Transferred from Another TLS": {
        "لا": "No",
        "لا ": "No",
        "نعم": "Yes",
    },
    
    "Disability or Chronic Illness": {
        "لا": "No",
        "لا ": "No",
        "نعم": "Yes",
    },
    
    "First-Degree Relative Loss": {
        "لا": "No loss",
        "لا ": "No loss",
        "نعم /الاب": "Father lost",
    },
    
    # Academic Level Classifications
    "Math Level": {
        'Beginning': 'Level 1',
        'Developing': 'Level 2',
        'proficient': 'Level 3',
        'Excellent': 'Level 4',
    },
    
    "Arabic Level": {
        'Beginning': 'Level 1',
        'Developing': 'Level 2',
        'proficient': 'Level 3',
        'Excellent': 'Level 4',
    },
    
    "Arabic Reading Category": {
        'Beginning': 'Level 1',
        'Developing': 'Level 2',
        'proficient': 'Level 3',
        'Excellent': 'Level 4',
    },
    
    "English Reading & Vocabulary Category": {
        'Beginning': 'Level 1',
        'Developing': 'Level 2',
        'proficient': 'Level 3',
        'Excellent': 'Level 4',
    },
    
    "English Level": {
        'Beginning': 'Level 1',
        'Developing': 'Level 2',
        'proficient': 'Level 3',
        'Excellent': 'Level 4',
    },
    
    "Average Category": {  # Only map the CATEGORY column, not the numeric score
        'Beginning': 'Level 1',
        'Developing': 'Level 2',
        'Advanced': 'Level 3',
        'Excellent': 'Level 4',
    },
    # Note: "Average Score" is NOT here because it's numeric, not categorical
}

# Normalization functions 
def get_subject_display_name(subject_col: str) -> str:
    return subject_col.replace(" Score", "").strip()

def get_mapped_column_name(subject_col: str) -> str:
    return f"{subject_col} mapped"

def normalize_column_name(col_name: str) -> str:
    if ' / ' in col_name:
        col_name = col_name.split(' / ')[0].strip()
    
    col_name = unicodedata.normalize("NFKC", str(col_name))
    col_name = col_name.replace("\u200f", "").replace("\u200e", "").replace("\u0640", "")
    col_name = re.sub(r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED]", "", col_name)
    col_name = re.sub(r'\s+', ' ', col_name).strip()
    col_name = (col_name.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
                .replace("ى", "ي").replace("ئ", "ي").replace("ؤ", "و").replace("ة", "ه"))
    
    return col_name

def validate_config():
    errors = []
    from pathlib import Path
    
    if not Path(Old_file).exists():
        errors.append(f"Excel file not found: {Old_file}")
    if not sheets:
        errors.append("No sheets configured")
    for csv_path, label in sheets:
        if not label:
            errors.append(f"Empty label for sheet: {csv_path}")
    for t in THRESHOLDS:
        if not (0 <= t <= TOTAL_MARK):
            errors.append(f"Invalid threshold: {t} (must be 0-{TOTAL_MARK})")
    if not SUBJECT_FINAL_COLS:
        errors.append("No subject columns configured")
    
    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(f"- {e}" for e in errors))
    
    return True

PALETTE = {
    "bg":             "#0A1220",   # Deep navy (darker than your current background)
    "surface":        "#0F1829",   # Slightly brighter navy for cards / panels
    "panel":          "#152033",   # Tertiary navy for contrast
    "border":         "#1F2C44",   # Subtle desaturated blue border
    "grid":           "#1B263A",   # Grid / separators
    
    "text":           "#E8EEF6",   # Soft white-blue (easier on eyes)
    "muted":          "#A7B4C8",   # Muted text / placeholders
    
    # NEW BLUE SYSTEM
    "primary":        "#4BA3F1",   # Main sky-blue (buttons, highlights)
    "primary2":       "#2D8BDC",   # Darker blue for active/hovers
    "accent":         "#7CC7FF",   # Light accent blue (logo-style dot)
    "glow":           "#3BA0FF",   # Glow/highlight
    
    "accent_hover":   "#9AD8FF",
    "accent_pressed": "#5FB4E8",
}

