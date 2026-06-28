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

def load_sentiment_pipeline():
    return pipeline("sentiment-analysis", model=MODEL_NAME, tokenizer=MODEL_NAME)

def add_sentiment(df: pd.DataFrame, text_col: str = "clean_text", batch_size: int = 32) -> pd.DataFrame:
    sentiment_pipeline = load_sentiment_pipeline()
    df = df.copy()

    texts = df[text_col].fillna("").tolist()
    # truncate very long comments to avoid token limit errors 
    texts = [ t[:1000] for t in texts]

    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size] 
        batch_result = sentiment_pipeline(batch, truncation=True, max_length=510)
        results.extend(batch_result)
        print(f"  Processed {min(i + batch_size, len(texts))}/{len(texts)}")

    df['sentiment_label'] = [ r['label'] for r in results]  # positive/negative/neutral
    df['sentiment_score'] = [ r['score'] for r in results]  # confidence
 
    return df
    

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

    df = pd.read_csv(os.path.join(DATA_DIR, "cleaned_comments.csv"))
    df = add_sentiment(df)

    out_path = os.path.join(DATA_DIR, "comments_with_sentiment.csv")
    df.to_csv(out_path, index=False)

    print(f"Saved sentiment results to {out_path}")
    print(df["sentiment_label"].value_counts())
