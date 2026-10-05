import csv
import os

FILE_NAME = "expenses.csv"

def initialize_file():
    if not os.path.exists(FILE_NAME):
        with open(FILE_NAME, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["amount", "category"])

def add_expense():
    while True:
        try:
            amount = float(input("Enter expense amount: "))
            if amount <= 0:
                print("Amount must be greater than 0.")
                continue
            break
        except ValueError:
            print("Please enter a valid number.")

    category = input("Enter category: ").strip()

    if not category:
        print("Category cannot be empty.")
        return

    with open(FILE_NAME, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([amount, category])

    print("Expense added successfully!\n")

def view_expenses():
    try:
        with open(FILE_NAME, "r", newline="") as file:
            reader = csv.reader(file)
            next(reader)

            total = 0
            print("\n----- Expenses -----")
            for row in reader:
                amount = float(row[0])
                category = row[1]
                print(f"{category}: Rs {amount:.2f}")
                total += amount

            print(f"\nTotal expense: Rs {total:.2f}\n")

    except FileNotFoundError:
        print("Expense file not found.")

def main():
    initialize_file()

    while True:
        print("======== Expense Tracker ========")
        print("1. Add Expense")
        print("2. View Expenses")
        print("3. Exit")

        choice = input("Enter choice number: ")

        if choice == "1":
            add_expense()
        elif choice == "2":
            view_expenses()
        elif choice == "3":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please select 1, 2, or 3.")

if __name__ == "__main__":
    main()
