import pandas as pd

# Load dataset
df = pd.read_csv(r"C:\Users\Laptops Galaxy\titanic\train.csv")

# Initial check
print(df.head())
df.info()
print(df.isnull().sum())

# Handle missing Age values using median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Extract Deck from Cabin
df["Deck"] = df["Cabin"].str[0]
df["Deck"] = df["Deck"].fillna("Unknown")

# Remove Cabin column
df.drop("Cabin", axis=1, inplace=True)

# Handle missing Embarked values using mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Final check
print(df.isnull().sum())
