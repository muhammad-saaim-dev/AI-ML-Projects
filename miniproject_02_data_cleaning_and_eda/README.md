# 🚢📊 Mini Project 2: Titanic Data Cleaning & Retail Sales EDA

Hey everyone! For **Day 13 (Week 2)** of my AI/ML roadmap, I built this mini-project to master essential **Pandas** data wrangling, missing value strategies, and business exploratory data analysis (EDA).

This project is divided into two focused parts:
1. **Part A:** Titanic Dataset Cleaning & Preprocessing (`titanic_cleaning.py`)
2. **Part B:** Retail Sales Dataset Exploratory Data Analysis (`sales_analysis.py`)

---

## 📂 File Structure
```text
miniproject_02_data_cleaning_and_eda/
│
├── titanic.csv              # Raw Kaggle Titanic dataset (891 rows)
├── titanic_cleaned.csv      # Processed dataset ready for modeling
├── titanic_cleaning.py      # Data cleaning & imputation script
├── sales_dataset.csv        # Retail sales records
├── sales_analysis.py        # Sales analytics & revenue aggregation script
└── README.md                # Project documentation
```

---

## 🚢 Part A: Titanic Dataset Cleaning (`titanic_cleaning.py`)

Data cleaning is the most crucial foundation of machine learning. Raw real-world data contains missing entries, noisy strings, and sparse attributes.

### Key Operations Implemented:
1. **Age Imputation:** Handled 177 missing age records by filling them with the **statistical median (28.0 years)**. Using the median protects against extreme skewness or outliers.
2. **Deck Feature Extraction:** Intelligently extracted the passenger's deck level from the complex `Cabin` string (e.g. `C85` -> `C`). Missing deck entries were labeled as `"Unknown"`.
3. **Dropping Sparse Columns:** Dropped the raw `Cabin` column (which was over 77% null) to avoid noise.
4. **Mode Imputation for Categorical Features:** Handled missing embarkation ports using the statistical **mode** (`'S'` for Southampton).
5. **Output Export:** Automatically outputs the clean, zero-null DataFrame to `titanic_cleaned.csv`.

---

## 📈 Part B: Retail Sales Dataset Analysis (`sales_analysis.py`)

A practical business analytics script focusing on revenue metrics, volume breakdown, and temporal trends:

### Key Metrics & Insights:
- **Total Revenue:** ~$189,620 across 748 units sold.
- **Top Product by Revenue:** Laptops ($84,150) and Smartphones ($55,200) drove the majority of gross sales.
- **Top Product by Volume:** Coffee Makers (142 units) and Office Chairs (103 units).
- **Category Leader:** Electronics accounted for over 75% of total sales volume.
- **Regional Distribution:** Balanced demand across East ($52.3k), South ($47.7k), North ($45.0k), and West ($44.5k).
- **Time-Series Parsing:** Parsed transaction dates to evaluate month-over-month performance.

---

## 🏃 How to Run Locally

```bash
cd miniproject_02_data_cleaning_and_eda

# Run Part A (Titanic Cleaning)
python titanic_cleaning.py

# Run Part B (Sales Analysis)
python sales_analysis.py
```

---
*Created as part of Day 13 of my Applied AI/ML Roadmap.*
