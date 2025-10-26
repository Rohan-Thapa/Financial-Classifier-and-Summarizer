# transaction_inference.py
"""
Simple zero-shot Income vs. Expense classifier using a pretrained HuggingFace model.

Input CSV ('transactions.csv') must have columns:
    Transaction (str), Amount (numeric)

This script will:
1. Load the CSV.
2. Classify each Transaction as 'Income' or 'Expense' using a zero-shot model.
3. Append 'PredictedType' and save to 'transactions_classified.csv'.
"""
import pandas as pd
from transformers import pipeline

# 1. Load data
df = pd.read_csv('transactions.csv')
if 'Transaction' not in df.columns:
    raise ValueError("CSV must contain 'Transaction' column.")

# 2. Initialize zero-shot classifier
# Requires internet to download model on first run
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)
labels = ["Income", "Expense"]

# 3. Classify each transaction
preds = []
for desc in df['Transaction'].astype(str):
    result = classifier(desc, labels)
    preds.append(result['labels'][0])

# 4. Append and save
df['PredictedType'] = preds
df.to_csv('transactions_classified.csv', index=False)
print("Classified transactions saved to 'transactions_classified.csv'.")
