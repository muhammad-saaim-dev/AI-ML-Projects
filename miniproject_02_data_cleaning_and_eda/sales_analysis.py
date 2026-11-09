"""
Mini-Project 2 (Part B): Retail Sales Dataset Analysis & EDA
Author: Muhammad Saaim

Covers practical Pandas business analytics:
1. Data inspection, duplicate checks, and missing value verification
2. Datetime parsing and monthly trend aggregations
3. Product, Category, and Regional performance breakdowns
4. Revenue, volume, and unit economics calculations
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "sales_dataset.csv")

print("=" * 55)
print("     PART B: RETAIL SALES EXPLORATORY DATA ANALYSIS")
print("=" * 55)

# Load dataset
df = pd.read_csv(CSV_FILE)

# Basic inspection
print("First 5 records:")
print(df.head())
print(f"\nDataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nDataset Info:")
df.info()

# Check missing and duplicate values
print("\nMissing values per column:")
print(df.isnull().sum())
print("Duplicate rows count:", df.duplicated().sum())

# Convert Date to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Total quantity sold and total sales
total_qty = df["Quantity"].sum()
total_rev = df["Sales"].sum()
print("\n" + "-" * 55)
print(f"Overall Total Quantity Sold: {total_qty:,}")
print(f"Overall Total Sales Revenue: ${total_rev:,.2f}")
print("-" * 55)

# Product-wise sales
print("\nTop Best-Selling Products by Revenue ($):")
product_sales = df.groupby("Product")["Sales"].sum().sort_values(ascending=False)
for prod, val in product_sales.items():
    print(f"  - {prod:<16}: ${val:>10,.2f}")

# Category-wise sales
print("\nCategory-wise Revenue Breakdown ($):")
category_sales = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
for cat, val in category_sales.items():
    print(f"  - {cat:<16}: ${val:>10,.2f}")

# Region-wise sales
print("\nRegional Revenue Share ($):")
region_sales = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)
for reg, val in region_sales.items():
    print(f"  - {reg:<16}: ${val:>10,.2f}")

# Product-wise quantity sold
print("\nTop Products by Volume (Units Sold):")
product_quantity = df.groupby("Product")["Quantity"].sum().sort_values(ascending=False)
for prod, qty in product_quantity.items():
    print(f"  - {prod:<16}: {qty:>4} units")

# Average product price
print("\nAverage Item Price:")
avg_price = df.groupby("Product")["Price"].mean().sort_values(ascending=False)
for prod, prc in avg_price.items():
    print(f"  - {prod:<16}: ${prc:>7.2f}")

# Monthly sales
df["Month"] = df["Date"].dt.month_name()
print("\nMonthly Sales Trend:")
monthly_sales = df.groupby("Month", sort=False)["Sales"].sum()
for month, val in monthly_sales.items():
    print(f"  - {month:<12}: ${val:>10,.2f}")

print("=" * 55 + "\n")
