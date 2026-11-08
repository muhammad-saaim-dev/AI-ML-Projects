"""
Week 3: Exploratory Data Analysis (EDA) on Titanic Dataset
Using Pandas, SQLite (SQL), and Seaborn
-----------------------------------------------------------
This project demonstrates:
1. Data Ingestion & Preprocessing with Pandas
2. Relational Database Modeling & Analytical Queries with SQLite (SQL)
3. Publication-quality Data Visualizations with Seaborn & Matplotlib
4. Analytical insights on historical passenger survival patterns
"""

import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def setup_database(csv_path: str, db_path: str) -> sqlite3.Connection:
    """
    Loads Titanic CSV dataset into an SQLite database table 'passengers'.
    """
    print("=" * 60)
    print("1. LOADING DATASET & INITIALIZING SQL DATABASE")
    print("=" * 60)

    df = pd.read_csv(csv_path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.\n")

    conn = sqlite3.connect(db_path)
    df.to_sql("passengers", conn, if_exists="replace", index=False)
    print(f"Data successfully inserted into SQLite table 'passengers' at '{db_path}'.\n")
    return conn


def run_sql_analysis(conn: sqlite3.Connection):
    """
    Executes analytical SQL queries to answer core business/EDA questions.
    """
    print("=" * 60)
    print("2. EXECUTING SQL ANALYTICAL QUERIES")
    print("=" * 60)

    # Query 1: Overall Demographics & Survival Rate
    q1 = """
    SELECT
        COUNT(*) AS total_passengers,
        SUM(Survived) AS survivors,
        COUNT(*) - SUM(Survived) AS fatalities,
        ROUND(AVG(Survived) * 100.0, 2) AS overall_survival_rate_pct,
        ROUND(AVG(Fare), 2) AS avg_fare_paid
    FROM passengers;
    """
    print("[SQL Query 1] Overall Summary & Survival Rate:")
    res1 = pd.read_sql_query(q1, conn)
    print(res1.to_string(index=False))
    print("-" * 60)

    # Query 2: Survival by Gender and Passenger Class
    q2 = """
    SELECT
        Pclass AS passenger_class,
        Sex AS gender,
        COUNT(*) AS passenger_count,
        SUM(Survived) AS survivors,
        ROUND(AVG(Survived) * 100.0, 2) AS survival_rate_pct
    FROM passengers
    GROUP BY Pclass, Sex
    ORDER BY Pclass ASC, survival_rate_pct DESC;
    """
    print("\n[SQL Query 2] Survival Rate by Passenger Class and Gender:")
    res2 = pd.read_sql_query(q2, conn)
    print(res2.to_string(index=False))
    print("-" * 60)

    # Query 3: Age Demographic Tiers using CASE WHEN
    q3 = """
    SELECT
        CASE
            WHEN Age IS NULL THEN 'Unknown'
            WHEN Age < 13 THEN 'Child (0-12)'
            WHEN Age < 20 THEN 'Teenager (13-19)'
            WHEN Age < 60 THEN 'Adult (20-59)'
            ELSE 'Senior (60+)'
        END AS age_group,
        COUNT(*) AS total_passengers,
        SUM(Survived) AS survivors,
        ROUND(AVG(Survived) * 100.0, 2) AS survival_rate_pct
    FROM passengers
    GROUP BY age_group
    ORDER BY survival_rate_pct DESC;
    """
    print("\n[SQL Query 3] Survival by Age Group (CASE WHEN):")
    res3 = pd.read_sql_query(q3, conn)
    print(res3.to_string(index=False))
    print("-" * 60)

    # Query 4: Family Structure Impact (SibSp + Parch)
    q4 = """
    SELECT
        (SibSp + Parch + 1) AS family_size,
        CASE
            WHEN (SibSp + Parch) = 0 THEN 'Solo Traveler'
            WHEN (SibSp + Parch) BETWEEN 1 AND 3 THEN 'Small Family (2-4)'
            ELSE 'Large Family (5+)'
        END AS family_category,
        COUNT(*) AS total_passengers,
        SUM(Survived) AS survivors,
        ROUND(AVG(Survived) * 100.0, 2) AS survival_rate_pct
    FROM passengers
    GROUP BY family_category
    ORDER BY survival_rate_pct DESC;
    """
    print("\n[SQL Query 4] Survival Rate by Family Structure:")
    res4 = pd.read_sql_query(q4, conn)
    print(res4.to_string(index=False))
    print("-" * 60)

    # Query 5: Port of Embarkation Survival
    q5 = """
    SELECT
        CASE Embarked
            WHEN 'C' THEN 'Cherbourg'
            WHEN 'Q' THEN 'Queenstown'
            WHEN 'S' THEN 'Southampton'
            ELSE 'Unknown'
        END AS embark_port,
        COUNT(*) AS total_passengers,
        ROUND(AVG(Fare), 2) AS avg_fare,
        ROUND(AVG(Survived) * 100.0, 2) AS survival_rate_pct
    FROM passengers
    GROUP BY embark_port
    ORDER BY survival_rate_pct DESC;
    """
    print("\n[SQL Query 5] Survival Rate and Average Fare by Port of Embarkation:")
    res5 = pd.read_sql_query(q5, conn)
    print(res5.to_string(index=False))
    print("=" * 60)


def generate_visualizations(csv_path: str, output_dir: str):
    """
    Generates high-resolution EDA figures using Seaborn & Matplotlib.
    """
    print("\n3. GENERATING SEABORN VISUALIZATIONS")
    print("=" * 60)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Clean / derive helper columns for visualization
    df["Survival_Status"] = df["Survived"].map({0: "Perished", 1: "Survived"})
    df["Class_Label"] = df["Pclass"].map({1: "1st Class", 2: "2nd Class", 3: "3rd Class"})
    df["Family_Size"] = df["SibSp"] + df["Parch"] + 1

    # Set seaborn style theme
    sns.set_theme(style="whitegrid", font_scale=1.05)

    # -------------------------------------------------------------
    # Plot 1: Survival Rate by Gender and Passenger Class
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 5))
    ax1 = sns.barplot(
        data=df,
        x="Class_Label",
        y="Survived",
        hue="Sex",
        palette={"male": "#4A90E2", "female": "#E94E77"},
        errorbar=None,
        order=["1st Class", "2nd Class", "3rd Class"]
    )
    plt.title("Survival Rate by Passenger Class and Gender", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Passenger Class", fontweight="bold")
    plt.ylabel("Survival Rate (Proportion)", fontweight="bold")
    plt.ylim(0, 1.05)
    for p in ax1.patches:
        height = p.get_height()
        if height > 0:
            ax1.annotate(f"{height:.1%}",
                         (p.get_x() + p.get_width() / 2., height),
                         ha='center', va='bottom', fontsize=10, xytext=(0, 3),
                         textcoords='offset points')
    plt.legend(title="Gender", frameon=True)
    plt.tight_layout()
    p1_path = os.path.join(output_dir, "01_survival_by_gender_class.png")
    plt.savefig(p1_path, dpi=300)
    plt.close()
    print(f"Saved: {p1_path}")

    # -------------------------------------------------------------
    # Plot 2: Age Distribution by Survival Status (Histogram + KDE)
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 5))
    sns.histplot(
        data=df.dropna(subset=["Age"]),
        x="Age",
        hue="Survival_Status",
        kde=True,
        bins=30,
        palette={"Survived": "#2ECC71", "Perished": "#E74C3C"},
        alpha=0.45
    )
    plt.title("Age Distribution: Survived vs Perished Passengers", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Passenger Age", fontweight="bold")
    plt.ylabel("Count of Passengers", fontweight="bold")
    plt.tight_layout()
    p2_path = os.path.join(output_dir, "02_age_distribution.png")
    plt.savefig(p2_path, dpi=300)
    plt.close()
    print(f"Saved: {p2_path}")

    # -------------------------------------------------------------
    # Plot 3: Ticket Fare Distribution across Passenger Classes
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 5))
    sns.boxplot(
        data=df,
        x="Class_Label",
        y="Fare",
        hue="Survival_Status",
        palette={"Survived": "#2ECC71", "Perished": "#E74C3C"},
        order=["1st Class", "2nd Class", "3rd Class"],
        showfliers=False  # Omit extreme outliers to clearly see interquartile ranges
    )
    plt.title("Fare Distribution by Ticket Class and Survival Status", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Passenger Class", fontweight="bold")
    plt.ylabel("Fare Paid (£)", fontweight="bold")
    plt.tight_layout()
    p3_path = os.path.join(output_dir, "03_fare_by_class_boxplot.png")
    plt.savefig(p3_path, dpi=300)
    plt.close()
    print(f"Saved: {p3_path}")

    # -------------------------------------------------------------
    # Plot 4: Family Size vs Survival Rate
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 5))
    family_stats = df.groupby("Family_Size")["Survived"].agg(["mean", "count"]).reset_index()
    ax4 = sns.barplot(
        data=family_stats,
        x="Family_Size",
        y="mean",
        hue="Family_Size",
        palette="Blues_d",
        legend=False
    )
    plt.title("Survival Rate by Total Family Size (SibSp + Parch + 1)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Family Size on Board", fontweight="bold")
    plt.ylabel("Survival Rate", fontweight="bold")
    plt.ylim(0, 1.0)
    for p in ax4.patches:
        height = p.get_height()
        if height > 0:
            ax4.annotate(f"{height:.1%}",
                         (p.get_x() + p.get_width() / 2., height),
                         ha='center', va='bottom', fontsize=9, xytext=(0, 3),
                         textcoords='offset points')
    plt.tight_layout()
    p4_path = os.path.join(output_dir, "04_family_size_survival.png")
    plt.savefig(p4_path, dpi=300)
    plt.close()
    print(f"Saved: {p4_path}")

    # -------------------------------------------------------------
    # Plot 5: Correlation Heatmap of Numeric Attributes
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    numeric_df = df[["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "Family_Size"]].copy()
    corr_matrix = numeric_df.corr()
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.75,
        cbar_kws={"label": "Pearson Correlation"}
    )
    plt.title("Correlation Matrix of Numeric Features", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    p5_path = os.path.join(output_dir, "05_correlation_heatmap.png")
    plt.savefig(p5_path, dpi=300)
    plt.close()
    print(f"Saved: {p5_path}")
    print("=" * 60)
    print("All visualizations successfully generated and saved!\n")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(base_dir, "titanic.csv")
    db_file = os.path.join(base_dir, "titanic.db")
    plots_dir = os.path.join(base_dir, "plots")

    # Step 1: Create SQLite DB & Load CSV
    conn = setup_database(csv_file, db_file)

    try:
        # Step 2: Run Analytical SQL Queries
        run_sql_analysis(conn)

        # Step 3: Generate Seaborn Visualizations
        generate_visualizations(csv_file, plots_dir)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
