import os
from transformers import pipeline
import pandas as pd

# Directory where the pretrained model is stored
MODEL_DIR = '../trained_models/transaction_type'
LABELS = ["Income", "Expense"]


def load_classifier():
    """
    Loading the zero-shot classifier pipeline from the local directory.
    """
    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError(
            f"Model directory '{MODEL_DIR}' not found. Run train_model.py first."
        )
    return pipeline(
        "zero-shot-classification",
        model=MODEL_DIR,
        tokenizer=MODEL_DIR
    )


def classify_df(df: pd.DataFrame, classifier=None) -> pd.DataFrame:
    """
    Classifying the transactions in a DataFrame as Income or Expense. Practicing code for the model by Rohan Thapa.

    Args:
        df: DataFrame with column 'Transaction' (string).
        classifier: Optional zero-shot pipeline. If None, loaded automatically.

    Returns:
        DataFrame copy with new 'Type' column.
    """
    if 'Transaction' not in df.columns:
        raise ValueError("DataFrame must contain a 'Transaction' column.")
    if classifier is None:
        classifier = load_classifier()

    predictions = []
    for desc in df['Transaction'].astype(str):
        result = classifier(desc, candidate_labels=LABELS)
        predictions.append(result['labels'][0])

    df_out = df.copy()
    df_out['Type'] = predictions
    return df_out

df = pd.read_csv('transactions.csv')
output = classify_df(df)
print(output)

'''
The output of the following code is,

                                         Transaction         Category   Amount     Type
0                  Paid NPR 350 for NTC mobile topup     Mobile Topup      350  Expense
1              Sent Rs 2000 to Sita Bank for tuition        Education     2000  Expense
2  Going to Pokhara on bus at the ticket price of...   Transportation      350  Expense
3            Dinner at the Hilton Hotel for NPR 1500             Food     1500  Expense
4  Buying a new t-shirt for the party at price of...         Shopping      950  Expense
5        Going to office on taxi with fare of Rs 250   Transportation      250  Expense
6                  Paid NPR 250 for NTC mobile topup     Mobile Topup      250  Expense
7              Paid for online course fee of Rs. 280        Education      280  Expense
8                Paid NPR 150 for Ncell mobile topup     Mobile Topup      150  Expense
9                        Got the salary of NRP 65000           Salary    65000   Income

'''