import pandas as pd

# Load dataset
df = pd.read_csv("sales_dataset.csv")

# Basic inspection
print(df.head())
print(df.shape)
df.info()

# Check missing values
print(df.isnull().sum())

# Check duplicate rows
print(df.duplicated().sum())

# Convert Date to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Total quantity sold and total sales
print("Total Quantity Sold:", df["Quantity"].sum())
print("Total Sales:", df["Sales"].sum())

# Product-wise sales
product_sales = df.groupby("Product")["Sales"].sum()
print("Product-wise sales:")
print(product_sales.sort_values(ascending=False))

# Category-wise sales
category_sales = df.groupby("Category")["Sales"].sum()
print("Category-wise sales:")
print(category_sales.sort_values(ascending=False))

# Region-wise sales
region_sales = df.groupby("Region")["Sales"].sum()
print("Region-wise sales:")
print(region_sales.sort_values(ascending=False))

# Product-wise quantity sold
product_quantity = df.groupby("Product")["Quantity"].sum()
print("Product-wise quantity sold:")
print(product_quantity.sort_values(ascending=False))

# Average product price
avg_price = df.groupby("Product")["Price"].mean()
print("Average product price:")
print(avg_price.sort_values(ascending=False))

# Monthly sales
df["Month"] = df["Date"].dt.month
monthly_sales = df.groupby("Month")["Sales"].sum()
print("Monthly sales:")
print(monthly_sales)
