"""
Step 2: Clean and preprocess the raw Youtube comments.
 
Covers core NLP fundamentals:
- lowercasing
- removing URLs, markdown, special characters
- removing deleted/removed/bot comments
- tokenization
- stopword removal (kept as a SEPARATE column - don't destroy raw text,
  topic models like BERTopic actually want lightly-cleaned text, not
  aggressively stripped tokens)
 
pip install nltk pandas --break-system-packages
"""

import pandas as pd
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langdetect import detect, LangDetectException
 
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

STOPWORDS = set(stopwords.words("english"))

def is_english(text: str) -> bool:
    """Filter to English-only comments, since the sentiment model is English-only."""
    if not isinstance(text, str) or len(text.strip()) < 3:
        return False
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False


def basic_clean(text: str) -> str:
    """ Light cleaning: use this version is used for Topic Modeling"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|wwww\.\S+", " ", text) # Remove  URLs
    text = re.sub(r"\[.*?\]\(.*?\)", " ", text)    # Remove markdoen links
    text = re.sub(r"[^a-z0-9\s.,!?']", " ", text)   # Keep Basic information
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_and_remove_stopwords(text: str) -> list:
    """Heavier cleaning: use this version for word-frequency analysis."""
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and t not in STOPWORDS]
    return tokens


def is_removed_or_bot(text: str) -> bool:
    if not isinstance(text, str):
        return True
    low = text.strip().lower()
    if low in ("[deleted]", "[removed]", ""):
        return True
    if "i am a bot" in low or "this action was performed automatically" in low:
        return True
    return False


def preprocessed(df : pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[~df['comment_text'].apply(is_removed_or_bot)]
    df = df[df['comment_text'].str.len() > 10]              # drop very short noise comments

    df['clean_text'] = df['comment_text'].apply(basic_clean)

    df = df[df['clean_text'].apply(is_english)]
    df['tokens']     = df['clean_text'].apply(tokenize_and_remove_stopwords)
    df['token_count']= df['tokens'].apply(len)

    df = df[ df['token_count'] > 3 ]     # need at least a few real words

    date_col = "created_utc" if "created_utc" in df.columns else "published_at" # created_utc is for Reddit and published_at is for Youtube
    df[date_col] = pd.to_datetime(df[date_col])
    df['date'] = df[date_col].dt.date

    return df.reset_index(drop=True)




if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
    os.makedirs(DATA_DIR, exist_ok=True)

    raw = pd.read_csv(os.path.join(DATA_DIR, "raw_comments.csv"))
    cleaned = preprocessed(raw)
    out_path = os.path.join(DATA_DIR, "cleaned_comments.csv")

    cleaned.to_csv(out_path, index=False)
    print(f"Cleaned {len(raw)} -> {len(cleaned)} comments, Saved to {out_path}")

