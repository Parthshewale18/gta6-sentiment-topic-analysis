"""
Step 4: Discover what people are actually talking about using BERTopic.
 
BERTopic clusters comments into topics automatically using embeddings,
then extracts representative keywords per topic. No manual labeling needed.
 
pip install bertopic pandas --break-system-packages
(bertopic pulls in sentence-transformers, umap, hdbscan automatically)
"""

import pandas as pd
from bertopic import BERTopic

def run_topic_model(df : pd.DataFrame, text_col : str = "clean_text"):
    docs = df[text_col].fillna("").tolist()
    topic_model = BERTopic(min_topic_size=15, verbose=True)  # Each topic must contain at least 15 comments.

    topics, probs = topic_model.fit_transform(docs)

    df = df.copy()
    df['topic_id'] = topics
    df['topic_prob'] = probs

    return df, topic_model


def label_topics(topic_model : BERTopic, df : pd.DataFrame) -> pd.DataFrame:
    """Attach human-readable topic keywords to each row."""
    topic_info = topic_model.get_topic_info()
    id_to_label = dict(zip(topic_info['Topic'], topic_info['Name']))

    df = df.copy()
    df["topic_label"] = df["topic_id"].map(id_to_label)
    return df


if __name__ == "__main__":
    import os
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
 
    df = pd.read_csv(os.path.join(DATA_DIR, "comments_with_sentiment.csv"))
    df, topic_model = run_topic_model(df)
    df = label_topics(topic_model, df)
 
    out_path = os.path.join(DATA_DIR, "comments_with_topics.csv")
    df.to_csv(out_path, index=False)
 
    topic_model.save(os.path.join(DATA_DIR, "bertopic_model"), serialization="safetensors")
 
    print(f"Saved to {out_path}")
    print("\nTop topics found:")
    print(topic_model.get_topic_info().head(15))