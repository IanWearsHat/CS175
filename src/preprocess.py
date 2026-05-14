import re
import pandas as pd


def clean_tweet(text):
    """Cleans tweet text by removing URLs, mentions, hashtags, and special characters."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # remove URLs
    text = re.sub(r"@\w+", "", text)  # remove @mentions
    text = re.sub(r"#\w+", "", text)  # remove #hashtags
    text = re.sub(r"[^a-z\s]", "", text)  # remove punctuation / numbers
    text = re.sub(r"\s+", " ", text).strip()  # collapse whitespace
    return text


def load_data(train_path, val_path):
    """Loads training and validation data from CSV files."""
    df_train = pd.read_csv(
        train_path,
        header=None,
        names=["id", "entity", "sentiment", "text"],
    )
    df_val = pd.read_csv(
        val_path,
        header=None,
        names=["id", "entity", "sentiment", "text"],
    )
    print(f"Loaded {len(df_train):,} training rows")
    print(f"Loaded {len(df_val):,} validation rows\n")
    return df_train, df_val


def preprocess_data(df_train, df_val):
    """Cleans and filters the datasets."""
    print("Cleaning datasets...")
    df_train["clean_text"] = df_train["text"].apply(clean_tweet)
    df_train = df_train[df_train["clean_text"].str.len() > 0].reset_index(drop=True)

    df_val["clean_text"] = df_val["text"].apply(clean_tweet)
    df_val = df_val[df_val["clean_text"].str.len() > 0].reset_index(drop=True)

    print(
        f"After cleaning: {len(df_train):,} training, {len(df_val):,} validation rows remaining\n"
    )

    X_train, y_train = df_train["clean_text"], df_train["sentiment"]
    X_test, y_test = df_val["clean_text"], df_val["sentiment"]
    return X_train, y_train, X_test, y_test
