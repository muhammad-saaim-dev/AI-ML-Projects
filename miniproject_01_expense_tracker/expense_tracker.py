"""
Mini-Project 1: CLI Expense Tracker (CSV & JSON Persistence)
Author: Muhammad Saaim

A command-line expense tracker that:
- Prompts for expense amounts and categories with input validation
- Persists data to CSV (expenses.csv)
- Syncs and exports data to JSON (expenses.json)
- Displays itemized expenses and total expenditure
"""

import csv
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "expenses.csv")
JSON_FILE = os.path.join(BASE_DIR, "expenses.json")


def initialize_files():
    """Initializes CSV and JSON files if they don't already exist."""
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["amount", "category"])

    if not os.path.exists(JSON_FILE):
        with open(JSON_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, indent=4)


def add_expense():
    """Takes user input for expense, validates the numbers, and saves to CSV & JSON."""
    while True:
        try:
            amount = float(input("Enter expense amount: "))
            if amount <= 0:
                print("Amount must be greater than 0.")
                continue
            break
        except ValueError:
            print("Please enter a valid numeric value.")

    category = input("Enter category (e.g. Food, Rent, Travel): ").strip()
    if not category:
        print("Category cannot be empty.")
        return

    # 1. Append to CSV
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([amount, category])

    # 2. Sync to JSON
    expenses = []
    if os.path.exists(JSON_FILE):
        try:
            with open(JSON_FILE, "r", encoding="utf-8") as f:
                expenses = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            expenses = []

    expenses.append({"amount": amount, "category": category})

    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(expenses, f, indent=4)

    print(f"Expense of Rs {amount:.2f} ({category}) saved to CSV and JSON successfully!\n")


def view_expenses():
    """Reads and displays all expenses from the CSV file."""
    if not os.path.exists(CSV_FILE):
        print("No expenses found. Add some first!\n")
        return

    try:
        with open(CSV_FILE, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header

            total = 0.0
            rows = list(reader)

            if not rows:
                print("\nNo expenses logged yet.\n")
                return

            print("\n" + "-" * 35)
            print("          EXPENSE LOG")
            print("-" * 35)
            for row in rows:
                if len(row) >= 2:
                    amount = float(row[0])
                    cat = row[1]
                    print(f"{cat:<20}: Rs {amount:>9.2f}")
                    total += amount

            print("-" * 35)
            print(f"{'Total Spent':<20}: Rs {total:>9.2f}")
            print("-" * 35 + "\n")

    except FileNotFoundError:
        print("Expense file not found.\n")


def view_json_export():
    """Displays the JSON file contents to demonstrate structured data persistence."""
    if not os.path.exists(JSON_FILE):
        print("No JSON file found yet.\n")
        return

    with open(JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("\n--- JSON Storage View (`expenses.json`) ---")
    print(json.dumps(data, indent=4))
    print("-------------------------------------------\n")


def main():
    initialize_files()

    while True:
        print("======== Expense Tracker ========")
        print("1. Add Expense")
        print("2. View Expenses (Summary)")
        print("3. View JSON Storage")
        print("4. Exit")

        choice = input("Enter choice (1-4): ").strip()

        if choice == "1":
            add_expense()
        elif choice == "2":
            view_expenses()
        elif choice == "3":
            view_json_export()
        elif choice == "4":
            print("Exiting Expense Tracker. Have a great day!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.\n")


if __name__ == "__main__":
    main()
