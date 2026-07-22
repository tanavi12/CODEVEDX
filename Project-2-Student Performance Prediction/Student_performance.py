# ============================================================
#  Student Performance Prediction System
#  Author : Tanavi Toti
#  Internship : CodeVedX - AI/ML Batch 2026
#  Project 2  : ML + Data Analysis + Visualization
#  (lightly refactored version)
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

warnings.filterwarnings("ignore")

# ---------- File path ----------
DATA_FILE = "student_data.csv"

# Columns used as model inputs (kept in one place so it's easy to extend later)
FEATURE_COLUMNS = [
    "attendance_percent",
    "math_marks",
    "science_marks",
    "english_marks",
    "study_hours_per_day",
    "sleep_hours",
    "extracurricular",
]

DIVIDER = "=" * 60


# ============================================================
# SECTION 1 - Data Loading & Cleaning
# ============================================================

def load_dataset():
    """Load CSV, fill missing numeric values with the column median, return the DataFrame."""
    if not os.path.exists(DATA_FILE):
        print(f"  [ERROR] '{DATA_FILE}' not found. Please place it in the same folder.")
        return None

    df = pd.read_csv(DATA_FILE)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            df[col] = df[col].fillna(df[col].median())
            print(f"  [INFO] Filled {missing_count} missing value(s) in '{col}' with the median.")

    return df


def print_dataset_overview(df):
    """Display basic info about the dataset (menu option 1)."""
    print(f"\n{DIVIDER}")
    print("  DATASET OVERVIEW")
    print(DIVIDER)
    print(f"  Total students  : {len(df)}")
    print(f"  Total features  : {len(df.columns)}")
    print(f"  Missing values  : {df.isnull().sum().sum()}")
    print(f"\n  Columns: {list(df.columns)}")
    print("\n  First 5 rows:")
    print(df.head().to_string(index=False))
    print()


# ============================================================
# SECTION 2 - Input Helpers
# ============================================================

def read_float(prompt, min_val=0.0, max_val=9999.0):
    """Read a validated float from the user within [min_val, max_val]."""
    while True:
        try:
            value = float(input(prompt))
            if min_val <= value <= max_val:
                return value
            print(f"  Please enter a value between {min_val} and {max_val}.")
        except ValueError:
            print("  Invalid input - please enter a number.")


def read_int(prompt, min_val=0, max_val=9999):
    """Read a validated integer from the user within [min_val, max_val]."""
    while True:
        try:
            value = int(input(prompt))
            if min_val <= value <= max_val:
                return value
            print(f"  Please enter a number between {min_val} and {max_val}.")
        except ValueError:
            print("  Invalid input - please enter a whole number.")


# ============================================================
# SECTION 3 - Exploratory Data Analysis (EDA)
# ============================================================

def run_eda(df):
    """Menu option 2 - descriptive stats and quick insights."""
    print(f"\n{DIVIDER}")
    print("  EXPLORATORY DATA ANALYSIS")
    print(DIVIDER)

    print("\n  Descriptive Statistics:")
    print(df.describe().round(2).to_string())

    print("\n  Average scores by subject:")
    for subject_col in ["math_marks", "science_marks", "english_marks"]:
        print(f"     {subject_col:<20} : {df[subject_col].mean():.2f}")

    print("\n  Final grade distribution:")
    grade_bins = [0, 40, 55, 70, 85, 100]
    grade_labels = ["Fail (<40)", "D (40-55)", "C (55-70)", "B (70-85)", "A (85+)"]
    df["grade_band"] = pd.cut(df["final_grade"], bins=grade_bins, labels=grade_labels)
    print(df["grade_band"].value_counts().to_string())
    df.drop(columns=["grade_band"], inplace=True)

    print("\n  Correlation with final_grade:")
    numeric_df = df.select_dtypes(include=[np.number])
    correlations = numeric_df.corr()["final_grade"].drop("final_grade").sort_values(ascending=False)
    for feature_name, corr_value in correlations.items():
        bar = "#" * int(abs(corr_value) * 20)
        print(f"     {feature_name:<25} : {corr_value:+.3f}  {bar}")
    print()


# ============================================================
# SECTION 4 - Data Visualization
# ============================================================

def plot_charts(df):
    """Menu option 3 - generate 4 charts and save as PNG."""
    print("\n  Generating visualizations... please wait.")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Student Performance Analysis", fontsize=16, fontweight="bold", y=1.01)
    plt.subplots_adjust(hspace=0.4, wspace=0.35)

    # ---- Chart 1: Study Hours vs Final Grade (scatter) ----
    ax1 = axes[0, 0]
    scatter = ax1.scatter(
        df["study_hours_per_day"], df["final_grade"],
        c=df["attendance_percent"], cmap="viridis", alpha=0.75, s=60
    )
    plt.colorbar(scatter, ax=ax1, label="Attendance %")
    ax1.set_xlabel("Study Hours per Day")
    ax1.set_ylabel("Final Grade")
    ax1.set_title("Study Hours vs Final Grade\n(color = Attendance %)")
    ax1.grid(True, linestyle="--", alpha=0.4)

    # ---- Chart 2: Average subject marks bar chart ----
    ax2 = axes[0, 1]
    subject_cols = ["math_marks", "science_marks", "english_marks"]
    subject_averages = [df[col].mean() for col in subject_cols]
    bar_colors = ["#028090", "#02C39A", "#0D3349"]
    bars = ax2.bar(["Math", "Science", "English"], subject_averages, color=bar_colors, width=0.5)
    for bar, avg in zip(bars, subject_averages):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                  f"{avg:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Average Marks")
    ax2.set_title("Average Marks by Subject")
    ax2.set_ylim(0, 100)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    # ---- Chart 3: Attendance vs Final Grade (scatter with trend line) ----
    ax3 = axes[1, 0]
    ax3.scatter(df["attendance_percent"], df["final_grade"],
                color="#028090", alpha=0.65, s=55)
    trend_coeffs = np.polyfit(df["attendance_percent"], df["final_grade"], 1)
    trend_fn = np.poly1d(trend_coeffs)
    x_line = np.linspace(df["attendance_percent"].min(), df["attendance_percent"].max(), 100)
    ax3.plot(x_line, trend_fn(x_line), color="#F96167", linewidth=2, linestyle="--", label="Trend")
    ax3.set_xlabel("Attendance %")
    ax3.set_ylabel("Final Grade")
    ax3.set_title("Attendance % vs Final Grade")
    ax3.legend()
    ax3.grid(True, linestyle="--", alpha=0.4)

    # ---- Chart 4: Correlation heatmap ----
    ax4 = axes[1, 1]
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    sns.heatmap(
        corr_matrix, ax=ax4, annot=True, fmt=".2f",
        cmap="coolwarm", linewidths=0.5,
        annot_kws={"size": 7}, square=True
    )
    ax4.set_title("Feature Correlation Heatmap")
    ax4.tick_params(axis="x", rotation=45, labelsize=7)
    ax4.tick_params(axis="y", rotation=0, labelsize=7)

    plt.tight_layout()
    chart_path = "student_analysis_charts.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n  Charts saved as '{chart_path}'\n")


# ============================================================
# SECTION 5 - Feature Engineering & ML Model
# ============================================================

def build_feature_matrix(df):
    """Create feature matrix X and target vector y."""
    X = df[FEATURE_COLUMNS].values
    y = df["final_grade"].values
    return X, y


def train_and_compare_models(df):
    """
    Train Linear Regression and Random Forest, print a comparison table,
    and return the model with the higher R^2 score.
    """
    X, y = build_feature_matrix(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    candidate_models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    }

    print(f"\n{DIVIDER}")
    print("  MODEL TRAINING & EVALUATION")
    print(DIVIDER)
    print(f"  Training samples : {len(X_train)}")
    print(f"  Testing  samples : {len(X_test)}")
    print(DIVIDER)
    print(f"  {'Model':<22} | {'MAE':>6} | {'RMSE':>6} | {'R2 Score':>9}")
    print(f"  {'-' * 52}")

    best_model = None
    best_r2 = -999
    best_model_name = ""

    for model_name, model in candidate_models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        print(f"  {model_name:<22} | {mae:>6.2f} | {rmse:>6.2f} | {r2:>9.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_model_name = model_name

    print(DIVIDER)
    print(f"  Best model: {best_model_name} (R2 = {best_r2:.4f})\n")

    if best_model_name == "Random Forest":
        importances = best_model.feature_importances_
        print("  Feature Importances:")
        for feature_name, importance in sorted(zip(FEATURE_COLUMNS, importances), key=lambda x: -x[1]):
            bar = "#" * int(importance * 40)
            print(f"     {feature_name:<25} : {importance:.4f}  {bar}")
        print()

    return best_model


def grade_band_for_score(score):
    """Map a numeric score to a (label, emoji) grade band tuple."""
    if score >= 85:
        return "A (Excellent)", "*"
    if score >= 70:
        return "B (Good)", "+"
    if score >= 55:
        return "C (Average)", "~"
    if score >= 40:
        return "D (Below Average)", "!"
    return "F (Fail)", "x"


def predict_for_new_student(model):
    """Menu option 5 - predict a student's final grade from user input."""
    print("\n--- Enter Student Details for Prediction ---")

    attendance = read_float("  Attendance percentage (0-100)  : ", 0, 100)
    math_marks = read_float("  Math marks (0-100)             : ", 0, 100)
    science_marks = read_float("  Science marks (0-100)          : ", 0, 100)
    english_marks = read_float("  English marks (0-100)          : ", 0, 100)
    study_hours = read_float("  Study hours per day (0-24)     : ", 0, 24)
    sleep_hours = read_float("  Sleep hours per day (0-24)     : ", 0, 24)
    has_extracurricular = read_int("  Extracurricular activities (0=No, 1=Yes): ", 0, 1)

    student_features = np.array([[
        attendance, math_marks, science_marks, english_marks,
        study_hours, sleep_hours, has_extracurricular
    ]])

    predicted_grade = model.predict(student_features)[0]
    predicted_grade = max(0, min(100, predicted_grade))  # clamp to 0-100

    band_label, band_marker = grade_band_for_score(predicted_grade)

    print(f"\n{DIVIDER}")
    print("  PREDICTION RESULT")
    print(DIVIDER)
    print(f"  Predicted Final Grade : {predicted_grade:.2f} / 100")
    print(f"  Grade Band            : {band_label} [{band_marker}]")
    print(DIVIDER)

    print("\n  Suggestions:")
    if attendance < 75:
        print("     - Attendance is low - aim for at least 75%")
    if study_hours < 3:
        print("     - Increase study hours to at least 3 hours/day")
    if sleep_hours < 6:
        print("     - Getting more sleep improves performance")
    if math_marks < 50 or science_marks < 50 or english_marks < 50:
        print("     - Focus on weaker subjects with extra practice")
    if predicted_grade >= 85:
        print("     - Excellent performance! Keep it up")
    print()


# ============================================================
# SECTION 6 - Add New Student Record
# ============================================================

def add_new_student_record(df):
    """Menu option 4 - add a new student record and persist it to the CSV."""
    print("\n--- Add New Student Record ---")

    new_student_id = df["student_id"].max() + 1
    attendance = read_float("  Attendance percentage (0-100)  : ", 0, 100)
    math_marks = read_float("  Math marks (0-100)             : ", 0, 100)
    science_marks = read_float("  Science marks (0-100)          : ", 0, 100)
    english_marks = read_float("  English marks (0-100)          : ", 0, 100)
    study_hours = read_float("  Study hours per day            : ", 0, 24)
    sleep_hours = read_float("  Sleep hours per day            : ", 0, 24)
    has_extracurricular = read_int("  Extracurricular (0=No, 1=Yes)  : ", 0, 1)
    final_grade = read_float("  Final grade (0-100)            : ", 0, 100)

    new_record = {
        "student_id": int(new_student_id),
        "attendance_percent": attendance,
        "math_marks": math_marks,
        "science_marks": science_marks,
        "english_marks": english_marks,
        "study_hours_per_day": study_hours,
        "sleep_hours": sleep_hours,
        "extracurricular": has_extracurricular,
        "final_grade": final_grade,
    }

    df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)
    print(f"  Student {int(new_student_id)} added successfully! Total records: {len(df)}\n")
    return df


# ============================================================
# SECTION 7 - Main Menu
# ============================================================

def print_menu():
    print("""
+------------------------------------------------+
|   Student Performance Prediction System         |
|   CodeVedX AI/ML Internship - 2026               |
+------------------------------------------------+
|  1. View dataset overview                        |
|  2. Exploratory Data Analysis (EDA)               |
|  3. Visualize data (charts)                       |
|  4. Add new student record                        |
|  5. Train model & predict student grade           |
|  6. Exit                                          |
+------------------------------------------------+""")


def main():
    print("\n  Welcome to the Student Performance Prediction System!")

    df = load_dataset()
    if df is None:
        return

    trained_model = None

    while True:
        print_menu()
        choice = input("  Enter your choice (1-6): ").strip()

        if choice == "1":
            print_dataset_overview(df)

        elif choice == "2":
            run_eda(df)

        elif choice == "3":
            plot_charts(df)

        elif choice == "4":
            df = add_new_student_record(df)

        elif choice == "5":
            if trained_model is None:
                trained_model = train_and_compare_models(df)
            predict_for_new_student(trained_model)

        elif choice == "6":
            print("\n  Thank you! Goodbye\n")
            break

        else:
            print("\n  Invalid choice. Please enter a number between 1 and 6.\n")


if __name__ == "__main__":
    main()