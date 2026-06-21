import pandas as pd
import numpy as np

np.random.seed(42)

n = 5000

application = pd.DataFrame({
    "id": range(1, n + 1),
    "age": np.random.randint(21, 65, n),
    "gender": np.random.choice(["Male", "Female"], n),
    "education": np.random.choice(
        ["High School", "Diploma", "Bachelor", "Master"],
        n
    ),
    "employment_type": np.random.choice(
        ["Salaried", "Self-Employed", "Business"],
        n
    ),
    "annual_income": np.random.randint(20000, 250000, n),
    "loan_amount": np.random.randint(5000, 150000, n),
    "loan_term": np.random.choice([12, 24, 36, 48, 60], n),
    "purpose": np.random.choice(
        ["Home", "Car", "Education", "Personal"],
        n
    ),
    "home_ownership": np.random.choice(
        ["Owned", "Rented", "Mortgaged"],
        n
    ),
    "dependents": np.random.randint(0, 6, n)
})

print(application.head())
print(application.shape)

bureau = pd.DataFrame({
    "id": range(1, n + 1),
    "bureau_score": np.random.randint(300, 900, n),
    "num_of_accounts": np.random.randint(1, 12, n),
    "num_of_open_accounts": np.random.randint(1, 8, n),
    "total_debt": np.random.randint(1000, 200000, n),
    "delinquent_accounts": np.random.randint(0, 5, n)
})

print("\nBureau Dataset")
print(bureau.head())
print(bureau.shape)

previous = pd.DataFrame({
    "id": range(1, n + 1),
    "num_loans": np.random.randint(0, 10, n),
    "num_defaults": np.random.randint(0, 4, n),
    "total_loan_amount": np.random.randint(5000, 500000, n),
    "total_repaid_amount": np.random.randint(1000, 400000, n),
    "avg_dpd": np.random.randint(0, 90, n)
})

print("\nPrevious Loans Dataset")
print(previous.head())
print(previous.shape)

payments = pd.DataFrame({
    "id": range(1, n + 1),
    "months_on_book": np.random.randint(1, 60, n),
    "status": np.random.choice(
        ["Paid", "Late", "Default"],
        n,
        p=[0.75, 0.20, 0.05]
    ),
    "dpd": np.random.randint(0, 120, n),
    "payment_amount": np.random.randint(500, 15000, n)
})

print("\nPayments Dataset")
print(payments.head())
print(payments.shape)

credit = pd.DataFrame({
    "id": range(1, n + 1),
    "num_cards": np.random.randint(1, 5, n),
    "credit_limit": np.random.randint(10000, 300000, n),
    "utilization_ratio": np.random.uniform(0.05, 1.0, n),
    "max_utilization_ratio": np.random.uniform(0.20, 1.0, n),
    "late_payments": np.random.randint(0, 12, n)
})

print("\nCredit Card Dataset")
print(credit.head())
print(credit.shape)

risk_score = (
    previous["num_defaults"] * 4 +
    bureau["delinquent_accounts"] * 3 +
    previous["avg_dpd"] * 0.15 +
    credit["late_payments"] * 0.8 +
    credit["utilization_ratio"] * 10 +
    (bureau["total_debt"] / application["annual_income"]) * 5
)

default_12m = (
    risk_score >
    np.percentile(risk_score, 80)
).astype(int)

labels = pd.DataFrame({
    "id": range(1, n + 1),
    "default_12m": default_12m
})

print("\nLabels Dataset")
print(labels.head())
print(labels["default_12m"].value_counts())

test = application.sample(
    n=1000,
    random_state=42
).copy()

print("\nTest Dataset")
print(test.shape)

application.to_csv(
    "data/application.csv",
    index=False
)

bureau.to_csv(
    "data/bureau.csv",
    index=False
)

previous.to_csv(
    "data/previous_loans.csv",
    index=False
)

payments.to_csv(
    "data/payments.csv",
    index=False
)

credit.to_csv(
    "data/credit_card.csv",
    index=False
)

labels.to_csv(
    "data/train_labels.csv",
    index=False
)

test.to_csv(
    "data/test.csv",
    index=False
)

print("\nAll datasets generated successfully!")