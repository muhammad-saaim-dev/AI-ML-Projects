# 🚀 Python & Machine Learning Roadmap Projects

Welcome to my **Python and Machine Learning Roadmap** repository! This repository serves as a personal portfolio and progress tracker as I journey through the world of Data Science, Machine Learning, and Applied Artificial Intelligence.

Here, I document real-world projects, scripts, and analyses built step-by-step — starting from foundational Python CLI applications and moving towards advanced statistical analysis and machine learning from scratch.

---

## 🛠️ Technologies & Libraries Used
- **Language:** Python 3.12+
- **Core Libraries:** NumPy, Pandas, SciPy, Seaborn, Matplotlib, Scikit-Learn
- **Built-in Modules:** `sqlite3`, `os`, `csv`, `json`, `datetime`
- **Core Competencies:**
  - Command-Line Interfaces (CLI) & Dual Storage (CSV & JSON)
  - Data Cleaning, Feature Extraction & Missing Value Imputation
  - Relational Database Modeling & SQL Analytics (`sqlite3`)
  - Statistical Hypothesis Testing (Welch's T-Test, One-Way ANOVA, Chi-Square, Normality)
  - Exploratory Data Analysis (EDA) & Publication-Quality Seaborn Visualizations
  - Machine Learning Algorithms from Scratch: Multiple Linear Regression & Batch Gradient Descent (NumPy)

---

## 📂 Projects Portfolio

### 1. 💰 [Mini Project 1: CLI Expense Tracker (`miniproject_01_expense_tracker/`)](miniproject_01_expense_tracker/)
A command-line expense tracker designed to manage and monitor daily expenses with robust validation.
- **Key Features:**
  - Automated input validation to prevent crashes from negative or non-numeric inputs.
  - Dual persistent data storage: records are automatically written to both CSV (`expenses.csv`) and structured JSON (`expenses.json`).
  - Itemized logging and running total calculation.
  - Dedicated [Mini Project 1 README](miniproject_01_expense_tracker/README.md).

### 2. 🚢📊 [Mini Project 2: Titanic Data Cleaning & Retail Sales EDA (`miniproject_02_data_cleaning_and_eda/`)](miniproject_02_data_cleaning_and_eda/)
A dual-part data preprocessing and analytics project tackling real-world data challenges using `pandas`.
- **Part A (Titanic Cleaning):** Median imputation for missing ages, intelligent deck extraction from cabin codes, dropping high-null features, and mode imputation for embarkation ports.
- **Part B (Sales Analysis):** Business exploratory analysis calculating revenue, volume by category/product, regional demand share, and monthly sales trajectories.
- Dedicated [Mini Project 2 README](miniproject_02_data_cleaning_and_eda/README.md).

### 3. 📊 [Mini Project 3: Titanic EDA Report with SQL & Seaborn (`miniproject_03_eda_sql/`)](miniproject_03_eda_sql/)
A hybrid analytical pipeline combining Pandas, an in-process SQLite relational database, and Seaborn statistical graphics.
- **Key Features:**
  - Ingests dataset into a local SQLite table (`passengers`).
  - Executes SQL queries utilizing multi-column `GROUP BY`, aggregations, and `CASE WHEN` conditional age brackets.
  - Generates 5 high-resolution (300 DPI) Seaborn charts covering class/gender divides, age curves, fare boxplots, and numeric correlations.
  - Dedicated [Mini Project 3 README](miniproject_03_eda_sql/README.md).

### 4. 📈 [Mini Project 4: Statistical EDA & NumPy Linear Regression (`miniproject_04_linear_regression/`)](miniproject_04_linear_regression/)
An end-to-end project on the Medical Insurance dataset combining rigorous statistical hypothesis testing with a from-scratch implementation of Multiple Linear Regression using pure NumPy and Batch Gradient Descent.
- **Key Features:**
  - **5 Statistical Tests:** Independent Two-Sample T-Test (Smokers vs Non-Smokers), One-Way ANOVA across 4 geographic regions, Chi-Square test of independence (Sex vs Smoker), Pearson correlation with p-values, and D'Agostino normality test.
  - **NumPy from Scratch:** Derivation and implementation of hypothesis $\hat{y} = Xw + b$, MSE cost function, and gradient descent update equations.
  - **Mathematical Parity:** Matches Scikit-Learn's analytical OLS solver with exact decimal precision ($R^2 = 0.7510$, RMSE = $6,234.61).
  - Dedicated [Mini Project 4 README](miniproject_04_linear_regression/README.md).

---

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/muhammad-saaim-dev/AI-ML-Projects.git
   cd AI-ML-Projects
   ```

2. **Install Dependencies:**
   ```bash
   pip install numpy pandas scipy matplotlib seaborn scikit-learn
   ```

3. **Run any project:**
   - **Mini Project 1:** `python miniproject_01_expense_tracker/expense_tracker.py`
   - **Mini Project 2 (Titanic):** `python miniproject_02_data_cleaning_and_eda/titanic_cleaning.py`
   - **Mini Project 2 (Sales):** `python miniproject_02_data_cleaning_and_eda/sales_analysis.py`
   - **Mini Project 3:** `python miniproject_03_eda_sql/eda_sql_report.py`
   - **Mini Project 4:** `python miniproject_04_linear_regression/linear_regression_scratch.py`

---
*⭐ Feel free to explore the code. More Machine Learning models, deep learning architectures, and end-to-end pipelines will be added as I progress further in my roadmap!*
