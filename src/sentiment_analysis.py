"""
Step 3: Sentiment analysis using a pretrained transformer.
 
We use cardiffnlp/twitter-roberta-base-sentiment-latest because it's
trained on social media text (similar style to youtube comments),
not formal text — this matters a lot for accuracy.
 
pip install transformers torch pandas --break-system-packages
"""

import pandas as pd
import os
from transformers import pipeline

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

def add_sentiment():
    pass

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

    df = pd.read_csv(os.path.join(DATA_DIR, "cleaned_comments.csv"))
    df = add_sentiment(df)

    out_path = os.path.join(DATA_DIR, "comments_with_sentiment.csv")
    df.to_csv(out_path, index=False)

    print(f"Saved sentiment results to {out_path}")
    print(df["sentiment_label"].value_counts())
