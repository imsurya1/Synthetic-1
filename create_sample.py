import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducible test data
np.random.seed(42)
random.seed(42)

# Generate realistic financial sample data with 30 columns and 15 rows
data = {
    'TransactionID': [f'TXN{str(i).zfill(6)}' for i in range(100001, 100016)],
    'AccountNumber': [f'ACC{random.randint(10000, 99999)}' for _ in range(15)],
    'CustomerID': [f'CUST{str(i).zfill(5)}' for i in range(50001, 50016)],
    'TransactionDate': [datetime(2024, 1, 1) + timedelta(days=random.randint(0, 90)) for _ in range(15)],
    'TransactionType': np.random.choice(['DEBIT', 'CREDIT', 'TRANSFER', 'ATM_WITHDRAWAL', 'DEPOSIT'], 15),
    'Amount': np.round(np.random.uniform(10.00, 5000.00, 15), 2),
    'Balance': np.round(np.random.uniform(500.00, 25000.00, 15), 2),
    'MerchantCategory': np.random.choice(['GROCERY', 'GAS', 'RESTAURANT', 'RETAIL', 'UTILITIES', 'HEALTHCARE'], 15),
    'MerchantName': [f'Merchant_{i}' for i in range(1, 16)],
    'CurrencyCode': ['USD'] * 15,
    'Channel': np.random.choice(['ONLINE', 'ATM', 'BRANCH', 'MOBILE'], 15),
    'City': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], 15),
    'State': np.random.choice(['NY', 'CA', 'IL', 'TX', 'AZ'], 15),
    'Country': ['USA'] * 15,
    'IsInternational': np.random.choice([True, False], 15),
    'RiskScore': np.round(np.random.uniform(0.1, 0.9, 15), 2),
    'FraudFlag': np.random.choice([True, False], 15, p=[0.1, 0.9]),
    'ProcessingTime': np.round(np.random.uniform(0.5, 5.0, 15), 1),
    'InterestRate': np.round(np.random.uniform(0.5, 8.5, 15), 2),
    'CreditLimit': np.round(np.random.uniform(1000, 50000, 15), 0),
    'AvailableCredit': np.round(np.random.uniform(500, 45000, 15), 0),
    'LastPaymentDate': [datetime(2024, 1, 1) + timedelta(days=random.randint(-30, 0)) for _ in range(15)],
    'LastPaymentAmount': np.round(np.random.uniform(50.00, 2000.00, 15), 2),
    'MonthlyIncome': np.round(np.random.uniform(3000, 15000, 15), 0),
    'CreditScore': np.random.randint(580, 850, 15),
    'AccountAge': np.random.randint(1, 120, 15),  # months
    'NumberOfTransactions': np.random.randint(1, 50, 15),
    'AverageMonthlySpend': np.round(np.random.uniform(500, 8000, 15), 2),
    'ProductType': np.random.choice(['CHECKING', 'SAVINGS', 'CREDIT_CARD', 'LOAN', 'INVESTMENT'], 15),
    'BranchCode': [f'BR{str(random.randint(100, 999)).zfill(3)}' for _ in range(15)]
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to Excel
df.to_excel('sample_financial_data.xlsx', index=False)
print("Sample financial data created successfully!")
print(f"Shape: {df.shape}")
print("\nFirst few rows:")
print(df.head(3))