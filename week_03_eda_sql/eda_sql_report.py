"""
Day 20 Mini-Project: Titanic EDA with Pandas, SQL (SQLite), and Seaborn
Author: Muhammad Saaim

The goal of this project is to explore the Kaggle Titanic dataset using:
- Pandas for quick data loading and wrangling
- SQL (via SQLite) to write analytical queries and slice the numbers
- Seaborn & Matplotlib to turn those numbers into clean, visual charts
"""

import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data_into_sql(csv_path: str, db_path: str) -> sqlite3.Connection:
    """Reads the Titanic CSV and dumps it into a local SQLite table so we can query it."""
    print("-> Reading dataset from:", csv_path)
    df = pd.read_csv(csv_path)
    print(f"-> Loaded {len(df)} rows and {len(df.columns)} columns.")

    # Connect to SQLite (creates the file if it doesn't exist)
    conn = sqlite3.connect(db_path)
    
    # Store dataframe as an SQL table named 'passengers'
    df.to_sql("passengers", conn, if_exists="replace", index=False)
    print("-> Data saved into SQLite table 'passengers' successfully.\n")
    return conn


def run_sql_queries(conn: sqlite3.Connection):
    """Runs a series of practical SQL queries to answer key questions about survival."""
    print("=" * 55)
    print("          STEP 1: SQL DATA EXPLORATION")
    print("=" * 55)

    # 1. Overall survival numbers
    q1 = """
    SELECT 
        COUNT(*) AS total_passengers,
        SUM(Survived) AS survivors,
        COUNT(*) - SUM(Survived) AS lost,
        ROUND(AVG(Survived) * 100, 2) AS survival_rate_pct,
        ROUND(AVG(Fare), 2) AS avg_fare
    FROM passengers;
    """
    print("\n[Query 1] What was the overall survival rate?")
    print(pd.read_sql_query(q1, conn).to_string(index=False))

    # 2. Gender and Class breakdown
    # Did women and first class passengers actually have higher survival rates?
    q2 = """
    SELECT 
        Pclass AS ticket_class,
        Sex AS gender,
        COUNT(*) AS total,
        SUM(Survived) AS survivors,
        ROUND(AVG(Survived) * 100, 2) AS survival_pct
    FROM passengers
    GROUP BY Pclass, Sex
    ORDER BY Pclass, survival_pct DESC;
    """
    print("\n[Query 2] Survival rates broken down by ticket class and gender:")
    print(pd.read_sql_query(q2, conn).to_string(index=False))

    # 3. Age groups using CASE WHEN
    # Grouping continuous age into sensible buckets
    q3 = """
    SELECT 
        CASE 
            WHEN Age IS NULL THEN 'Unknown'
            WHEN Age < 13 THEN 'Children (0-12)'
            WHEN Age < 20 THEN 'Teens (13-19)'
            WHEN Age < 60 THEN 'Adults (20-59)'
            ELSE 'Seniors (60+)'
        END AS age_group,
        COUNT(*) AS total_count,
        SUM(Survived) AS survivors,
        ROUND(AVG(Survived) * 100, 2) AS survival_pct
    FROM passengers
    GROUP BY age_group
    ORDER BY survival_pct DESC;
    """
    print("\n[Query 3] How did age factor into survival? (using CASE WHEN):")
    print(pd.read_sql_query(q3, conn).to_string(index=False))

    # 4. Family size impact
    # Family size = SibSp (siblings/spouse) + Parch (parents/children) + 1 (the passenger)
    q4 = """
    SELECT 
        (SibSp + Parch + 1) AS family_size,
        CASE 
            WHEN (SibSp + Parch) = 0 THEN 'Solo Traveler'
            WHEN (SibSp + Parch) BETWEEN 1 AND 3 THEN 'Small Family (2-4)'
            ELSE 'Large Family (5+)'
        END AS family_type,
        COUNT(*) AS passenger_count,
        SUM(Survived) AS survivors,
        ROUND(AVG(Survived) * 100, 2) AS survival_pct
    FROM passengers
    GROUP BY family_type
    ORDER BY survival_pct DESC;
    """
    print("\n[Query 4] Did traveling alone or with family make a difference?")
    print(pd.read_sql_query(q4, conn).to_string(index=False))

    # 5. Port of Embarkation
    # C = Cherbourg, Q = Queenstown, S = Southampton
    q5 = """
    SELECT 
        CASE Embarked
            WHEN 'C' THEN 'Cherbourg'
            WHEN 'Q' THEN 'Queenstown'
            WHEN 'S' THEN 'Southampton'
            ELSE 'Unknown'
        END AS port_of_embarkation,
        COUNT(*) AS passenger_count,
        ROUND(AVG(Fare), 2) AS avg_fare,
        ROUND(AVG(Survived) * 100, 2) AS survival_pct
    FROM passengers
    GROUP BY port_of_embarkation
    ORDER BY survival_pct DESC;
    """
    print("\n[Query 5] Survival and average fare based on boarding port:")
    print(pd.read_sql_query(q5, conn).to_string(index=False))
    print("\n" + "=" * 55)


def create_visualizations(csv_path: str, output_folder: str):
    """Generates clean Seaborn plots to visually back up our SQL numbers."""
    print("\n" + "=" * 55)
    print("          STEP 2: CREATING SEABORN PLOTS")
    print("=" * 55)

    os.makedirs(output_folder, exist_ok=True)
    df = pd.read_csv(csv_path)

    # Add easy-to-read labels for plotting
    df["Status"] = df["Survived"].map({0: "Perished", 1: "Survived"})
    df["Class"] = df["Pclass"].map({1: "1st Class", 2: "2nd Class", 3: "3rd Class"})
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    # Setting a clean aesthetic for the charts
    sns.set_theme(style="whitegrid")

    # -------------------------------------------------------------
    # Chart 1: Survival by Class and Gender
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        data=df,
        x="Class",
        y="Survived",
        hue="Sex",
        palette={"male": "#457b9d", "female": "#e63946"},
        errorbar=None,
        order=["1st Class", "2nd Class", "3rd Class"]
    )
    plt.title("Titanic Survival Rate by Class and Gender", fontsize=13, fontweight="bold", pad=10)
    plt.xlabel("Ticket Class")
    plt.ylabel("Survival Percentage")
    plt.ylim(0, 1.1)

    # Add percentage labels above each bar
    for p in ax.patches:
        val = p.get_height()
        if val > 0:
            ax.annotate(f"{val:.1%}", (p.get_x() + p.get_width() / 2, val),
                        ha='center', va='bottom', fontsize=10, xytext=(0, 3),
                        textcoords='offset points')

    plt.legend(title="Gender")
    plt.tight_layout()
    p1 = os.path.join(output_folder, "01_survival_by_gender_class.png")
    plt.savefig(p1, dpi=300)
    plt.close()
    print("Saved Chart 1:", p1)

    # -------------------------------------------------------------
    # Chart 2: Age Distribution (Who survived vs who didn't)
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=df.dropna(subset=["Age"]),
        x="Age",
        hue="Status",
        bins=25,
        kde=True,
        palette={"Survived": "#2a9d8f", "Perished": "#e76f51"},
        alpha=0.5
    )
    plt.title("Age Distribution: Survived vs Perished", fontsize=13, fontweight="bold", pad=10)
    plt.xlabel("Age (Years)")
    plt.ylabel("Passenger Count")
    plt.tight_layout()
    p2 = os.path.join(output_folder, "02_age_distribution.png")
    plt.savefig(p2, dpi=300)
    plt.close()
    print("Saved Chart 2:", p2)

    # -------------------------------------------------------------
    # Chart 3: Fare distribution by Class
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="Class",
        y="Fare",
        hue="Status",
        palette={"Survived": "#2a9d8f", "Perished": "#e76f51"},
        order=["1st Class", "2nd Class", "3rd Class"],
        showfliers=False  # Hide extreme outliers so the box shapes are visible
    )
    plt.title("Ticket Fare vs Survival across Classes", fontsize=13, fontweight="bold", pad=10)
    plt.xlabel("Ticket Class")
    plt.ylabel("Fare Paid (£)")
    plt.tight_layout()
    p3 = os.path.join(output_folder, "03_fare_by_class_boxplot.png")
    plt.savefig(p3, dpi=300)
    plt.close()
    print("Saved Chart 3:", p3)

    # -------------------------------------------------------------
    # Chart 4: Family size vs survival rate
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    family_summary = df.groupby("FamilySize")["Survived"].mean().reset_index()
    ax_fam = sns.barplot(
        data=family_summary,
        x="FamilySize",
        y="Survived",
        hue="FamilySize",
        palette="viridis",
        legend=False
    )
    plt.title("Survival Rate based on Total Family Size", fontsize=13, fontweight="bold", pad=10)
    plt.xlabel("Total Family Members on Board (including self)")
    plt.ylabel("Survival Rate")
    plt.ylim(0, 1.0)

    for p in ax_fam.patches:
        val = p.get_height()
        if val > 0:
            ax_fam.annotate(f"{val:.1%}", (p.get_x() + p.get_width() / 2, val),
                            ha='center', va='bottom', fontsize=9, xytext=(0, 2),
                            textcoords='offset points')

    plt.tight_layout()
    p4 = os.path.join(output_folder, "04_family_size_survival.png")
    plt.savefig(p4, dpi=300)
    plt.close()
    print("Saved Chart 4:", p4)

    # -------------------------------------------------------------
    # Chart 5: Correlation Heatmap
    # -------------------------------------------------------------
    plt.figure(figsize=(7, 5))
    numeric_cols = df[["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize"]].copy()
    corr = numeric_cols.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Between Numerical Features", fontsize=13, fontweight="bold", pad=10)
    plt.tight_layout()
    p5 = os.path.join(output_folder, "05_correlation_heatmap.png")
    plt.savefig(p5, dpi=300)
    plt.close()
    print("Saved Chart 5:", p5)

    print("\nAll charts successfully created in:", output_folder)


def main():
    base_path = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(base_path, "titanic.csv")
    db_file = os.path.join(base_path, "titanic.db")
    plots_dir = os.path.join(base_path, "plots")

    # 1. Load data and setup SQLite
    conn = load_data_into_sql(csv_file, db_file)

    try:
        # 2. Run SQL analysis
        run_sql_queries(conn)

        # 3. Create visual plots
        create_visualizations(csv_file, plots_dir)

    finally:
        conn.close()
        print("\n[Done] Analysis complete!")


if __name__ == "__main__":
    main()
