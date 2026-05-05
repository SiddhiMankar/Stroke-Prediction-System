"""
Data Quality Check Script
Purpose: Validate dataset cleanliness before EDA & modeling
Branch: eda-data-validation
"""

import pandas as pd


# ===============================
# 1. LOAD DATA
# ===============================

def load_data(path):
    df = pd.read_csv(path)
    print("Dataset loaded successfully.")
    return df


# ===============================
# 2. BASIC STRUCTURE CHECK
# ===============================

def check_structure(df):
    print("\n--- STRUCTURE CHECK ---")
    print("Shape:", df.shape)
    print("\nColumn Names:")
    print(df.columns.tolist())
    print("\nData Types:")
    print(df.dtypes)


# ===============================
# 3. DUPLICATE CHECK
# ===============================

def check_duplicates(df):
    duplicates = df.duplicated().sum()
    print("\n--- DUPLICATE CHECK ---")
    print(f"Number of duplicate rows: {duplicates}")


# ===============================
# 4. MISSING VALUES CHECK
# ===============================

def check_missing_values(df):
    print("\n--- MISSING VALUES CHECK ---")
    missing = df.isnull().sum()
    missing_percent = (df.isnull().sum() / len(df)) * 100

    summary = pd.DataFrame({
        "Missing_Count": missing,
        "Missing_Percentage": missing_percent
    })

    print(summary[summary["Missing_Count"] > 0])


# ===============================
# 5. NUMERIC RANGE CHECK
# ===============================

def check_numeric_ranges(df):
    print("\n--- NUMERIC RANGE CHECK ---")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    for col in numeric_cols:
        print(f"\nColumn: {col}")
        print("Min:", df[col].min())
        print("Max:", df[col].max())


# ===============================
# 6. TARGET VARIABLE CHECK
# ===============================

def check_target_distribution(df, target_column):
    print("\n--- TARGET DISTRIBUTION ---")
    print(df[target_column].value_counts())
    print("\nPercentage Distribution:")
    print(df[target_column].value_counts(normalize=True) * 100)


# ===============================
# MAIN EXECUTION
# ===============================

if __name__ == "__main__":
    
    DATA_PATH = r"C:\Projects\Stroke-awareness-TY-mini-project\data\stroke_cleaned_final.csv"
    TARGET_COLUMN = "stroke"  # change if needed
    
    df = load_data(DATA_PATH)
    
    check_structure(df)
    check_duplicates(df)
    check_missing_values(df)
    check_numeric_ranges(df)
    check_target_distribution(df, TARGET_COLUMN)

    print("\nData Quality Check Completed.")
