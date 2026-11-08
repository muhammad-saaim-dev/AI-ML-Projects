# 🚀 Python & Machine Learning Roadmap Projects

Welcome to my **Python and Machine Learning Roadmap** repository! This repository serves as a personal portfolio and progress tracker as I journey through the world of Data Science, Machine Learning, and Python programming. 

Here, I document the real-world projects, scripts, and analyses I build to strengthen my core concepts. This repository is structured to show my day-to-day progress, starting from foundational Python applications and moving towards advanced Data Science techniques.

---

## 🛠️ Technologies & Libraries Used
- **Language:** Python 3.x
- **Libraries:** Pandas, Seaborn, Matplotlib, built-in modules (`sqlite3`, `os`, `csv`, `datetime`)
- **Concepts Covered:** 
  - Command-Line Interfaces (CLI) & File Handling (CSV)
  - Data Cleaning & Missing Value Imputation
  - Relational Database Modeling & SQL Queries (`sqlite3`)
  - Statistical Visualizations & Exploratory Data Analysis (EDA)
  - Demographic Segmentation (`CASE WHEN`, `GROUP BY`)

---

## 📂 Current Projects Overview

### 1. 💰 Expense Tracker (`expense_tracker.py`)
A robust, CLI-based Python application designed to help users manage their daily expenses effectively.
- **Features:** 
  - Add new expenses with automated input validation to prevent crashes from invalid data types.
  - View a detailed list of all logged expenses and categories.
  - Automatically calculates and displays the total amount spent.
- **Technical Highlights:** Implements persistent data storage using Python's native `csv` and `os` modules. Data is safely written to and read from `expenses.csv`, ensuring expenses are not lost after closing the program.

### 2. 🚢 Titanic Dataset Cleaning (`titanic_cleaning.py`)
A comprehensive data preprocessing script tackling the famous Kaggle Titanic dataset. Data cleaning is the most crucial step in Machine Learning, and this script demonstrates essential techniques.
- **Key Operations:**
  - **Age Imputation:** Fills missing age values using the statistical `median` to avoid outliers skewing the data.
  - **Feature Extraction:** Intelligently extracts the `Deck` level from the complex `Cabin` string format.
  - **Handling Unknowns:** Replaces missing `Deck` values with "Unknown" and completely drops the highly sparse `Cabin` column.
  - **Mode Imputation:** Fills missing `Embarked` (port of embarkation) values using the most frequent value (`mode`).
- **Goal:** To produce a clean, ready-to-use dataset for future predictive modeling.

### 3. 📈 Sales Dataset Analysis (`sales_analysis.py`)
A detailed Exploratory Data Analysis (EDA) project focusing on a retail sales dataset to extract meaningful business insights using `pandas`.
- **Key Operations:**
  - **Data Inspection & Cleaning:** Checks dataset shape, data types, null values, and duplicate records.
  - **Type Conversion:** Accurately converts date strings into proper `datetime` objects for time-series analysis.
  - **Business Metrics Calculated:**
    - Overall Total Sales Revenue & Total Quantity Sold.
    - Product-wise performance (Best-selling products by volume and revenue).
    - Category-wise and Region-wise market share.
    - Average pricing analysis for individual items.
    - Monthly sales trends.

### 4. 📊 Week 3 Mini-Project: EDA Report using Pandas, SQL & Seaborn (`week_03_eda_sql/`)
A comprehensive, publication-ready Exploratory Data Analysis mini-project on the Kaggle Titanic dataset combining Pandas, SQLite relational database queries, and Seaborn statistical plots.
- **Key Highlights:**
  - Ingests dataset into an in-memory / local **SQLite database**.
  - Runs advanced SQL queries using `GROUP BY`, multi-column aggregations, and `CASE WHEN` conditional age tiers.
  - Produces high-resolution Seaborn figures (Class/Gender survival bar plots, Age distribution histograms + KDE, Fare boxplots, and numeric correlation heatmap).
  - Includes a dedicated [Week 3 README](week_03_eda_sql/README.md) with full query tables, findings, and charts.

---

## 🚀 How to Run Locally

To test these projects on your local machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/muhammad-saaim-dev/AI-ML-Projects.git
   cd AI-ML-Projects
   ```
2. **Install Dependencies:**
   Ensure you have Python installed. Then install Pandas, Seaborn, and Matplotlib:
   ```bash
   pip install pandas seaborn matplotlib
   ```
3. **Run the scripts:**
   - Expense Tracker: `python expense_tracker.py`
   - Titanic Cleaning: `python titanic_cleaning.py`
   - Sales Analysis: `python sales_analysis.py`
   - Week 3 EDA & SQL: `python week_03_eda_sql/eda_sql_report.py`

---
*⭐ Feel free to explore the code. More complex Machine Learning projects and deep learning architectures will be added as I progress further in my roadmap!*
