import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Directory to save the pretrained model and tokenizer
MODEL_DIR = '../trained_models/transaction_type'
BASE_MODEL = 'facebook/bart-large-mnli'


def train_and_save():
    if not os.path.isdir(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)
        # Downloading the model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL)
        # Saving the model locally
        tokenizer.save_pretrained(MODEL_DIR)
        model.save_pretrained(MODEL_DIR)
        print(f"Model downloaded and saved to '{MODEL_DIR}'")
    else:
        print(f"Model directory '{MODEL_DIR}' already exists. Skipping download.")


if __name__ == '__main__':
    train_and_save()