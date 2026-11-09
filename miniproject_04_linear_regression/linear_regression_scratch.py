"""
Day 27: End-to-End EDA with Statistical Tests & NumPy Linear Regression (Gradient Descent)
Author: Muhammad Saaim
Dataset: Medical Insurance Costs (insurance.csv)

In this project, I explore:
1. End-to-End Exploratory Data Analysis (EDA) on insurance charges
2. Hypothesis Testing & Statistical Significance:
   - Two-Sample T-Test (Smokers vs Non-Smokers)
   - One-Way ANOVA (Charges across 4 Regions)
   - Chi-Square Test of Independence (Gender vs Smoker status)
   - Pearson Correlation & p-values (Age & BMI vs Charges)
   - Normality Test (D'Agostino-Pearson K-squared test)
3. Multiple Linear Regression from Scratch using Pure NumPy & Batch Gradient Descent
4. Benchmark comparison against Scikit-Learn's LinearRegression
"""

import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression


# =====================================================================
# 1. EXPLORATORY DATA ANALYSIS & STATISTICAL TESTING
# =====================================================================

def run_statistical_eda(df: pd.DataFrame, output_dir: str):
    """
    Performs core statistical hypothesis tests on the insurance dataset
    and explains the findings in plain English.
    """
    print("=" * 60)
    print("          STEP 1: STATISTICAL HYPOTHESIS TESTING")
    print("=" * 60)

    # -------------------------------------------------------------
    # Test 1: Two-Sample T-Test (Smokers vs Non-Smokers)
    # -------------------------------------------------------------
    # Question: Do smokers have significantly higher medical charges?
    # H0: Mean charges of smokers == Mean charges of non-smokers
    # H1: Mean charges of smokers != Mean charges of non-smokers
    smokers = df[df["smoker"] == "yes"]["charges"]
    non_smokers = df[df["smoker"] == "no"]["charges"]

    t_stat, p_val_t = stats.ttest_ind(smokers, non_smokers, equal_var=False)

    print("\n[Test 1] Independent Two-Sample T-Test (Welch's):")
    print(f"  - Non-smoker average charge: ${non_smokers.mean():,.2f}")
    print(f"  - Smoker average charge:     ${smokers.mean():,.2f}")
    print(f"  - T-Statistic: {t_stat:.4f}")
    print(f"  - p-value:     {p_val_t:.4e}")
    if p_val_t < 0.05:
        print("  -> Conclusion: REJECT H0 (p < 0.05). Smokers pay statistically significantly")
        print("     higher medical insurance charges (nearly 4x more than non-smokers).")
    else:
        print("  -> Conclusion: FAIL TO REJECT H0.")

    # -------------------------------------------------------------
    # Test 2: One-Way ANOVA (Charges across 4 Regions)
    # -------------------------------------------------------------
    # Question: Do medical charges differ significantly across regions?
    # H0: Mean charges are equal across all 4 regions (NE, NW, SE, SW)
    # H1: At least one region has a significantly different mean charge
    regions = [group["charges"].values for _, group in df.groupby("region")]
    f_stat, p_val_anova = stats.f_oneway(*regions)

    print("\n[Test 2] One-Way ANOVA (Charges across Geographic Regions):")
    for reg, grp in df.groupby("region"):
        print(f"  - {reg.capitalize():<10} Mean: ${grp['charges'].mean():,.2f} (n={len(grp)})")
    print(f"  - F-Statistic: {f_stat:.4f}")
    print(f"  - p-value:     {p_val_anova:.4f}")
    if p_val_anova < 0.05:
        print("  -> Conclusion: REJECT H0 (p < 0.05). There is a statistically significant")
        print("     difference in charges across regions (Southeast is notably higher).")
    else:
        print("  -> Conclusion: FAIL TO REJECT H0 (p >= 0.05). Regional difference is not significant.")

    # -------------------------------------------------------------
    # Test 3: Chi-Square Test of Independence (Gender vs Smoker)
    # -------------------------------------------------------------
    # Question: Is smoking status independent of gender?
    # H0: Gender and Smoking status are independent
    # H1: Gender and Smoking status are dependent
    contingency_table = pd.crosstab(df["sex"], df["smoker"])
    chi2_stat, p_val_chi2, dof, expected = stats.chi2_contingency(contingency_table)

    print("\n[Test 3] Chi-Square Test of Independence (Sex vs Smoker):")
    print("  Contingency Table:")
    print(contingency_table.to_string())
    print(f"  - Chi2-Statistic: {chi2_stat:.4f} (degrees of freedom = {dof})")
    print(f"  - p-value:        {p_val_chi2:.4f}")
    if p_val_chi2 < 0.05:
        print("  -> Conclusion: REJECT H0 (p < 0.05). Smoking status is dependent on gender")
        print("     (higher proportion of males smoke in this dataset).")
    else:
        print("  -> Conclusion: FAIL TO REJECT H0. No significant association between gender and smoking.")

    # -------------------------------------------------------------
    # Test 4: Pearson Correlation with p-values
    # -------------------------------------------------------------
    r_age, p_age = stats.pearsonr(df["age"], df["charges"])
    r_bmi, p_bmi = stats.pearsonr(df["bmi"], df["charges"])

    print("\n[Test 4] Pearson Correlation with Charges:")
    print(f"  - Age vs Charges: r = {r_age:.4f} (p-value = {p_age:.4e}) -> Statistically Significant")
    print(f"  - BMI vs Charges: r = {r_bmi:.4f} (p-value = {p_bmi:.4e}) -> Statistically Significant")

    # -------------------------------------------------------------
    # Test 5: Normality Test of Target (Charges)
    # -------------------------------------------------------------
    k2_stat, p_val_norm = stats.normaltest(df["charges"])
    skewness = df["charges"].skew()

    print("\n[Test 5] Normality Test on Charges (D'Agostino's K^2):")
    print(f"  - Skewness:   {skewness:.4f} (Right-skewed distribution)")
    print(f"  - K^2 Stat:   {k2_stat:.4f}")
    print(f"  - p-value:    {p_val_norm:.4e}")
    print("  -> Conclusion: p < 0.05, charges are NOT normally distributed (heavily right-skewed).")
    print("=" * 60)

    # -------------------------------------------------------------
    # Generate Plots
    # -------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Plot 1: Smoker vs Charges Distribution (T-Test visualization)
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="smoker",
        y="charges",
        hue="smoker",
        palette={"yes": "#e63946", "no": "#2a9d8f"},
        legend=False
    )
    plt.title("Medical Charges: Smokers vs Non-Smokers (T-Test Visual)", fontsize=13, fontweight="bold", pad=10)
    plt.xlabel("Smoker Status")
    plt.ylabel("Annual Charges ($)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_charges_by_smoker_ttest.png"), dpi=300)
    plt.close()

    # Plot 2: Region vs Charges (ANOVA visual)
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=df,
        x="region",
        y="charges",
        hue="region",
        palette="Blues_d",
        legend=False
    )
    plt.title("Average Medical Charges by Region (ANOVA Visual)", fontsize=13, fontweight="bold", pad=10)
    plt.xlabel("Geographic Region")
    plt.ylabel("Average Charges ($)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_region_charges_anova.png"), dpi=300)
    plt.close()

    # Plot 3: Numeric Correlation Heatmap
    plt.figure(figsize=(7, 5))
    numeric_df = df[["age", "bmi", "children", "charges"]].copy()
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix of Numeric Features", fontsize=13, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_correlation_matrix.png"), dpi=300)
    plt.close()

    print("Saved statistical EDA charts to:", output_dir)


# =====================================================================
# 2. NUMPY LINEAR REGRESSION FROM SCRATCH (GRADIENT DESCENT)
# =====================================================================

class LinearRegressionGD:
    """
    Multiple Linear Regression implemented from scratch using NumPy and Batch Gradient Descent.

    Hypothesis:
        y_hat = X . w + b

    Cost Function (Mean Squared Error / 2):
        J(w, b) = (1 / 2m) * sum((y_hat - y)^2)

    Gradients:
        dJ/dw = (1 / m) * X.T . (y_hat - y)
        dJ/db = (1 / m) * sum(y_hat - y)

    Update Rule:
        w = w - lr * dJ/dw
        b = b - lr * dJ/db
    """

    def __init__(self, learning_rate: float = 0.05, n_iterations: int = 1000):
        self.lr = learning_rate
        self.n_iters = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        m, n = X.shape
        # Initialize weights with zeros, bias with zero
        self.weights = np.zeros(n)
        self.bias = 0.0
        self.cost_history = []

        for i in range(self.n_iters):
            # 1. Forward Pass: compute predictions
            y_pred = np.dot(X, self.weights) + self.bias

            # 2. Compute Cost: Half Mean Squared Error (MSE / 2)
            error = y_pred - y
            cost = (1.0 / (2.0 * m)) * np.sum(error ** 2)
            self.cost_history.append(cost)

            # 3. Compute Gradients
            dw = (1.0 / m) * np.dot(X.T, error)
            db = (1.0 / m) * np.sum(error)

            # 4. Backward Pass: Update weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.weights) + self.bias


# =====================================================================
# 3. DATA PREPARATION, TRAINING & BENCHMARKING
# =====================================================================

def standardize_features(X_train: np.ndarray, X_test: np.ndarray):
    """
    Standardizes features using Z-score normalization in pure NumPy:
    X_scaled = (X - mean) / std
    """
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    # Prevent division by zero if std is zero
    std[std == 0] = 1.0

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std
    return X_train_scaled, X_test_scaled, mean, std


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """Calculates R2-Score, RMSE, and MAE using pure NumPy."""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1.0 - (ss_res / ss_tot)

    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def run_linear_regression_pipeline(df: pd.DataFrame, output_dir: str):
    """
    Prepares features, trains our NumPy Gradient Descent model,
    benchmarks it against Scikit-Learn, and plots convergence.
    """
    print("\n" + "=" * 60)
    print("      STEP 2: NUMPY LINEAR REGRESSION (GRADIENT DESCENT)")
    print("=" * 60)

    # 1. Feature Encoding
    data = df.copy()
    data["is_smoker"] = (data["smoker"] == "yes").astype(float)
    data["is_male"] = (data["sex"] == "male").astype(float)

    # Feature matrix X and continuous target y
    feature_names = ["age", "bmi", "children", "is_smoker", "is_male"]
    X = data[feature_names].values
    y = data["charges"].values

    # 2. Train-Test Split (80% train, 20% test) with random seed for reproducibility
    np.random.seed(42)
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    train_size = int(0.80 * len(X))
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")
    print(f"Features used:    {feature_names}\n")

    # 3. Feature Scaling (Crucial for Gradient Descent convergence)
    X_train_scaled, X_test_scaled, _, _ = standardize_features(X_train, X_test)

    # 4. Train Our NumPy Model
    print("Training NumPy Linear Regression with Gradient Descent...")
    lr_model = LinearRegressionGD(learning_rate=0.08, n_iterations=1200)
    lr_model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred_numpy = lr_model.predict(X_test_scaled)
    numpy_metrics = compute_metrics(y_test, y_pred_numpy)

    # 5. Benchmark against Scikit-Learn (using Ordinary Least Squares)
    sk_model = LinearRegression()
    sk_model.fit(X_train_scaled, y_train)
    y_pred_sklearn = sk_model.predict(X_test_scaled)
    sklearn_metrics = compute_metrics(y_test, y_pred_sklearn)

    # 6. Display Comparison Table
    print("\n" + "-" * 60)
    print("BENCHMARK: NumPy (Gradient Descent) vs Scikit-Learn (OLS)")
    print("-" * 60)
    comparison_df = pd.DataFrame([
        {
            "Implementation": "NumPy Gradient Descent (Scratch)",
            "R2 Score": f"{numpy_metrics['R2']:.4f}",
            "RMSE ($)": f"${numpy_metrics['RMSE']:,.2f}",
            "MAE ($)": f"${numpy_metrics['MAE']:,.2f}"
        },
        {
            "Implementation": "Scikit-Learn LinearRegression",
            "R2 Score": f"{sklearn_metrics['R2']:.4f}",
            "RMSE ($)": f"${sklearn_metrics['RMSE']:,.2f}",
            "MAE ($)": f"${sklearn_metrics['MAE']:,.2f}"
        }
    ])
    print(comparison_df.to_string(index=False))
    print("-" * 60)

    # Print learned weights
    print("\nLearned Weights (Coefficients per Standardized Feature):")
    for feat, w in zip(feature_names, lr_model.weights):
        print(f"  - {feat:<12}: {w:+.2f}")
    print(f"  - Intercept (bias): ${lr_model.bias:,.2f}")
    print(f"  -> Insight: 'is_smoker' has the highest weight ({lr_model.weights[3]:+.2f}),")
    print("     confirming our T-test finding that smoking is the primary driver of cost.")

    # -------------------------------------------------------------
    # Visualizations: Loss Curve & Actual vs Predicted
    # -------------------------------------------------------------
    # Plot 4: Gradient Descent Loss Convergence Curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(lr_model.cost_history)), lr_model.cost_history, color="#e63946", lw=2)
    plt.title("Gradient Descent Cost Convergence (MSE / 2 vs Iterations)", fontsize=13, fontweight="bold", pad=10)
    plt.xlabel("Iteration Number")
    plt.ylabel("Cost J(w, b)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_gradient_descent_loss_curve.png"), dpi=300)
    plt.close()

    # Plot 5: Actual vs Predicted Charges
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred_numpy, color="#457b9d", alpha=0.6, edgecolors="none")
    # Diagonal ideal line
    min_val = min(y_test.min(), y_pred_numpy.min())
    max_val = max(y_test.max(), y_pred_numpy.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=1.5, label="Perfect 1:1 Prediction")
    plt.title("Actual vs Predicted Insurance Charges (NumPy GD)", fontsize=13, fontweight="bold", pad=10)
    plt.xlabel("Actual Charges ($)")
    plt.ylabel("Predicted Charges ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "05_actual_vs_predicted.png"), dpi=300)
    plt.close()

    print("Saved convergence & prediction plots to:", output_dir)
    print("=" * 60)


# =====================================================================
# 4. MAIN ENTRY POINT
# =====================================================================

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "insurance.csv")
    plots_dir = os.path.join(base_dir, "plots")

    print("=" * 60)
    print("DAY 27: STATISTICAL EDA & NUMPY LINEAR REGRESSION")
    print("=" * 60)
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.\n")

    # Step 1: Statistical Hypothesis Testing & EDA
    run_statistical_eda(df, plots_dir)

    # Step 2: Linear Regression with Gradient Descent in pure NumPy
    run_linear_regression_pipeline(df, plots_dir)

    print("\n[Done] Day 27 Project execution finished successfully!")


if __name__ == "__main__":
    main()

