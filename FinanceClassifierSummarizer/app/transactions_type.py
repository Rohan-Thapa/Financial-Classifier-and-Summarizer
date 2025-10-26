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
        DataFrame copy with new 'Type' column which is necessary for the model
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

# HandCrafted by Rohan Thapa.