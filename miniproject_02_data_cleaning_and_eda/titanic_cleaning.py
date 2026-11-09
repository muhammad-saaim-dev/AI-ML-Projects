"""
Mini-Project 2 (Part A): Titanic Dataset Data Cleaning & Preprocessing
Author: Muhammad Saaim

Covers core Pandas data cleaning fundamentals:
1. Missing value inspection and handling
2. Imputation of numerical features using statistical median
3. Imputation of categorical features using statistical mode
4. Feature extraction: Extracting 'Deck' from raw 'Cabin' strings
5. Dropping sparse and high-cardinality columns
"""

import os
import pandas as pd

# Portable path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(BASE_DIR, "titanic.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "titanic_cleaned.csv")

print("=" * 55)
print("     PART A: TITANIC DATA CLEANING & PREPROCESSING")
print("=" * 55)

# 1. Load dataset
df = pd.read_csv(INPUT_CSV)
print(f"Loaded raw dataset with {df.shape[0]} rows and {df.shape[1]} columns.\n")

print("--- Missing Values Before Cleaning ---")
print(df.isnull().sum()[df.isnull().sum() > 0])
print("-" * 55)

# 2. Handle missing Age values using median (robust against outliers)
age_median = df["Age"].median()
df["Age"] = df["Age"].fillna(age_median)
print(f"-> Filled missing Age values with median: {age_median:.1f}")

# 3. Extract Deck from Cabin
df["Deck"] = df["Cabin"].str[0]
df["Deck"] = df["Deck"].fillna("Unknown")
print("-> Extracted 'Deck' level from Cabin strings (missing labeled as 'Unknown').")

# 4. Remove sparse Cabin column
df = df.drop(columns=["Cabin"])
print("-> Dropped original 'Cabin' column.")

# 5. Handle missing Embarked values using mode (most frequent port)
embarked_mode = df["Embarked"].mode()[0]
df["Embarked"] = df["Embarked"].fillna(embarked_mode)
print(f"-> Filled missing Embarked values with mode: '{embarked_mode}'")

print("\n--- Missing Values After Cleaning ---")
print(df.isnull().sum())
print("-" * 55)

# Save cleaned dataset
df.to_csv(OUTPUT_CSV, index=False)
print(f"Cleaned dataset successfully saved to: {OUTPUT_CSV}")
print("=" * 55 + "\n")
