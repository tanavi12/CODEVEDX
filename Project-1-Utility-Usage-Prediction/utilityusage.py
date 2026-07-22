# Utility Usage Prediction Tool
# Author       : Tanavi Toti
# Internship   : CodeVedX – AI/ML Batch 2026
# Project 1    : Menu-driven console ML application
# ============================================================

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import csv
import os

# ---------- File path ----------
DATA_FILE = "utility_data.csv"


# ============================================================
# SECTION 1 – CSV Helpers
# ============================================================
def load_data():
    """Load CSV into a pandas DataFrame. Returns empty DataFrame if file missing."""
    if not os.path.exists(DATA_FILE):
        # Create a fresh file with headers only
        headers = ["month", "num_people", "temperature_celsius",
                   "appliances_count", "electricity_units",
                   "water_liters", "gas_units"]
        with open(DATA_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        print(f"[INFO] '{DATA_FILE}' not found – created a new empty file.")
        return pd.DataFrame(columns=headers)
    df = pd.read_csv(DATA_FILE)
    return df


def save_row(row: dict):
    """Append a single row dictionary to the CSV file."""
    file_exists = os.path.exists(DATA_FILE)
    with open(DATA_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()  # write header only once
        writer.writerow(row)


# ============================================================
# SECTION 2 – Input Helpers (with exception handling)
# ============================================================
def get_int(prompt, min_val=1, max_val=9999):
    """Ask user for an integer within [min_val, max_val]."""
    while True:
        try:
            value = int(input(prompt))
            if min_val <= value <= max_val:
                return value
            print(f"  Please enter a number between {min_val} and {max_val}.")
        except ValueError:
            print("  Invalid input – please enter a whole number.")


def get_float(prompt, min_val=0.0):
    """Ask user for a non-negative float."""
    while True:
        try:
            value = float(input(prompt))
            if value >= min_val:
                return value
            print(f"  Please enter a value >= {min_val}.")
        except ValueError:
            print("  Invalid input – please enter a number (e.g. 25.5).")


# ============================================================
# SECTION 3 – Core Features
# ============================================================
def add_usage_record():
    """Menu Option 1 – manually add a new monthly usage record."""
    print("\n--- Add New Usage Record ---")
    row = {
        "month": get_int("  Month (1-12)           : ", 1, 12),
        "num_people": get_int("  Number of people       : ", 1, 50),
        "temperature_celsius": get_float("  Avg temperature (°C)  : "),
        "appliances_count": get_int("  Number of appliances   : ", 1, 100),
        "electricity_units": get_float("  Electricity used (kWh): "),
        "water_liters": get_float("  Water used (litres)   : "),
        "gas_units": get_float("  Gas used (units)      : "),
    }
    save_row(row)
    print("  ✔ Record saved successfully!\n")


def update_last_record():
    """Menu Option 2 – overwrite the last row in the CSV."""
    df = load_data()
    if df.empty:
        print("\n  No records found. Please add data first.\n")
        return

    print(f"\n  Last record:\n{df.tail(1).to_string(index=False)}\n")
    print("  Enter updated values for the last record:")

    updated = {
        "month": get_int("  Month (1-12)           : ", 1, 12),
        "num_people": get_int("  Number of people       : ", 1, 50),
        "temperature_celsius": get_float("  Avg temperature (°C)  : "),
        "appliances_count": get_int("  Number of appliances   : ", 1, 100),
        "electricity_units": get_float("  Electricity used (kWh): "),
        "water_liters": get_float("  Water used (litres)   : "),
        "gas_units": get_float("  Gas used (units)      : "),
    }

    df.iloc[-1] = updated          # replace last row in memory
    df.to_csv(DATA_FILE, index=False)
    print("  ✔ Last record updated!\n")


def view_all_records():
    """Menu Option 3 – display the full dataset."""
    df = load_data()
    if df.empty:
        print("\n  No records available yet.\n")
        return

    print(f"\n{'='*60}")
    print("  ALL UTILITY RECORDS")
    print(f"{'='*60}")
    print(df.to_string(index=False))
    print(f"\n  Total records : {len(df)}\n")


def show_statistics():
    """Menu Option 4 – basic descriptive stats of the dataset."""
    df = load_data()
    if df.empty:
        print("\n  No data to analyse yet.\n")
        return

    print(f"\n{'='*60}")
    print("  DESCRIPTIVE STATISTICS")
    print(f"{'='*60}")
    print(df.describe().round(2).to_string())
    print(f"\n  Average electricity per person : "
          f"{(df['electricity_units'] / df['num_people']).mean():.2f} kWh")
    print(f"  Average water per person       : "
          f"{(df['water_liters'] / df['num_people']).mean():.2f} litres\n")


# ============================================================
# SECTION 4 – Machine Learning Prediction
# ============================================================
def train_and_predict():
    """
    Menu Option 5 – Train a Linear Regression model on the CSV data,
    evaluate it, then predict usage for user-provided inputs.
    Features  → month, num_people, temperature_celsius, appliances_count
    Targets   → electricity_units, water_liters, gas_units
    """
    df = load_data()
    if len(df) < 5:
        print("\n  Not enough data to train a model (need at least 5 records).\n")
        return

    # ---- Prepare features and labels ----
    feature_cols = ["month", "num_people", "temperature_celsius", "appliances_count"]
    target_cols = ["electricity_units", "water_liters", "gas_units"]
    X = df[feature_cols].values

    results = {}  # store one model per target

    print(f"\n{'='*60}")
    print("  MODEL TRAINING REPORT")
    print(f"{'='*60}")

    for target in target_cols:
        y = df[target].values
        # Split: 80% train, 20% test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[target] = model
        print(f"  {target:<22} | MAE: {mae:7.2f} | R² Score: {r2:.4f}")
    print(f"{'='*60}")

    # ---- Get prediction inputs ----
    print("\n  Enter details to predict your next month's usage:\n")
    month = get_int("  Month (1-12)           : ", 1, 12)
    num_people = get_int("  Number of people       : ", 1, 50)
    temperature = get_float("  Expected temperature (°C): ")
    appliances = get_int("  Number of appliances   : ", 1, 100)

    user_input = np.array([[month, num_people, temperature, appliances]])

    print(f"\n{'='*60}")
    print("  PREDICTED UTILITY USAGE")
    print(f"{'='*60}")

    labels = {
        "electricity_units": ("Electricity", "kWh"),
        "water_liters": ("Water      ", "litres"),
        "gas_units": ("Gas        ", "units"),
    }

    for target, model in results.items():
        prediction = model.predict(user_input)[0]
        prediction = max(0, prediction)  # never negative
        label, unit = labels[target]
        print(f"  {label} : {prediction:>8.2f} {unit}")
    print(f"{'='*60}\n")


# ============================================================
# SECTION 5 – Main Menu
# ============================================================
def print_menu():
    print("""
╔══════════════════════════════════════════╗
║    Utility Usage Prediction Tool         ║
║    CodeVedX AI/ML Internship – 2026      ║
╠══════════════════════════════════════════╣
║  1. Add new usage record                 ║
║  2. Update last record                   ║
║  3. View all records                     ║
║  4. Show statistics                      ║
║  5. Predict next month's usage (ML)      ║
║  6. Exit                                 ║
╚══════════════════════════════════════════╝
""")


def main():
    print("\n  Welcome to the Utility Usage Prediction Tool!")
    load_data()  # ensure file exists on startup

    while True:
        print_menu()
        choice = input("  Enter your choice (1-6): ").strip()

        if choice == "1":
            add_usage_record()
        elif choice == "2":
            update_last_record()
        elif choice == "3":
            view_all_records()
        elif choice == "4":
            show_statistics()
        elif choice == "5":
            train_and_predict()
        elif choice == "6":
            print("\n  Exiting prediction model. \n")
            break
        else:
            print("\n  Invalid choice. Re enter the code\n")


# ---- Entry point ----
if __name__ == "__main__":
    main()