# 💰 Mini Project 1: CLI Expense Tracker (CSV & JSON)

Hey everyone! As part of **Week 1** of my AI/ML roadmap, I built this command-line application to practice core Python fundamentals:
- Building interactive CLI interfaces with loop controls
- Robust user input validation and error handling (`try-except`)
- Persistent file storage using both **CSV** (`expenses.csv`) and **JSON** (`expenses.json`)

---

## 🛠️ Features
- **Add Expense:** Enter any positive expense amount and assign it to a category (e.g. Food, Groceries, Rent, Utilities).
- **Automated Validation:** Prevents program crashes from accidental string inputs or negative amounts.
- **Dual Persistence:** Automatically writes data to tabular CSV and structured JSON files without losing history when you close the program.
- **Summary & Totals:** Clean formatting displaying itemized expenditures and automatic calculation of total spending.

---

## 📂 File Structure
```text
miniproject_01_expense_tracker/
│
├── expense_tracker.py       # Main CLI application
├── expenses.csv             # Tabular storage
├── expenses.json            # JSON structured storage
└── README.md                # Project documentation
```

---

## 🏃 How to Run Locally

```bash
cd miniproject_01_expense_tracker
python expense_tracker.py
```

### Sample Session:
```text
======== Expense Tracker ========
1. Add Expense
2. View Expenses (Summary)
3. View JSON Storage
4. Exit
Enter choice (1-4): 1
Enter expense amount: 450
Enter category (e.g. Food, Rent, Travel): Groceries
Expense of Rs 450.00 (Groceries) saved to CSV and JSON successfully!
```

---
*Created as part of my Applied AI/ML Roadmap.*
