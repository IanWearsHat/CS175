import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline

# ── 1. Load Data ──────────────────────────────────────────────────────────────
df_train = pd.read_csv(
    "data/archive/twitter_training.csv",
    header=None,
    names=["id", "entity", "sentiment", "text"],
)
df_val = pd.read_csv(
    "data/archive/twitter_validation.csv",
    header=None,
    names=["id", "entity", "sentiment", "text"],
)

print(f"Loaded {len(df_train):,} training rows")
print(f"Loaded {len(df_val):,} validation rows\n")


# ── 2. Clean Tweets ───────────────────────────────────────────────────────────
def clean_tweet(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # remove URLs
    text = re.sub(r"@\w+", "", text)  # remove @mentions
    text = re.sub(r"#\w+", "", text)  # remove #hashtags
    text = re.sub(r"[^a-z\s]", "", text)  # remove punctuation / numbers
    text = re.sub(r"\s+", " ", text).strip()  # collapse whitespace
    return text


print("Cleaning datasets...")
df_train["clean_text"] = df_train["text"].apply(clean_tweet)
df_train = df_train[df_train["clean_text"].str.len() > 0].reset_index(drop=True)

df_val["clean_text"] = df_val["text"].apply(clean_tweet)
df_val = df_val[df_val["clean_text"].str.len() > 0].reset_index(drop=True)

print(
    f"After cleaning: {len(df_train):,} training, {len(df_val):,} validation rows remaining\n"
)

# ── 3. Assign Features / Labels ───────────────────────────────────────────────
X_train, y_train = df_train["clean_text"], df_train["sentiment"]
X_test, y_test = df_val["clean_text"], df_val["sentiment"]

# ── 4. Build Pipeline with Best Hyperparameters ───────────────────────────────
# Best params from GridSearchCV (3-fold CV, f1_weighted):
#   clf__C=15.0, clf__solver=saga
#   tfidf__max_features=50000, tfidf__ngram_range=(1,2), tfidf__sublinear_tf=True
# Best CV F1: 0.8517  |  Test F1: 0.8865  |  Test Accuracy: 0.8866
pipeline = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),  # unigrams + bigrams
                max_features=50_000,  # vocabulary cap
                sublinear_tf=True,  # log-scale term frequencies
                min_df=2,
            ),
        ),
        (
            "clf",
            LogisticRegression(
                C=1.0,
                solver="saga",
                max_iter=1000,
                n_jobs=-1,
            ),
        ),
    ]
)

# ── 5. Train ──────────────────────────────────────────────────────────────────
print("Training with best hyperparameters...")
pipeline.fit(X_train, y_train)
print("Done.\n")

# ── 6. Evaluate ───────────────────────────────────────────────────────────────
y_pred = pipeline.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

print(f"Accuracy      : {acc:.4f}")
print(f"F1 (weighted) : {f1:.4f}\n")
print("Per-class report:")
print(classification_report(y_test, y_pred))

# ── 7. Example Predictions ────────────────────────────────────────────────────
examples = [
    "ive never seen a more well-put together clothing line",
    "this game is gorgeously horrendous",
    "Won meaningless games down the stretch the one year they had an opportunity to lock in a top-5 pick. Now they're picking 7th. The Sacramento Kings franchise will forever be bad due to their own incompetence.",
    "Wemby is a generational talent.  He is going to be a top player of all time. He is going to become more and more aggressive, and get defended more and more physically. He, like Shaq before him, is going to need to learn to help referees, to protect himself, and to be his own enforcer. This elbow, though. This ain't it.",
]

print("=== Example Predictions ===")
cleaned_examples = [clean_tweet(t) for t in examples]
predictions = pipeline.predict(cleaned_examples)
probabilities = pipeline.predict_proba(cleaned_examples)
classes = pipeline.classes_

for tweet, pred, probs in zip(examples, predictions, probabilities):
    prob_str = "  ".join(f"{cls}={p:.2f}" for cls, p in zip(classes, probs))
    print(f"\nTweet      : {tweet}")
    print(f"Prediction : {pred}")
    print(f"Confidence : {prob_str}")


"""
Using grid search with
param_grid = {
    # TF-IDF
    "tfidf__ngram_range": [(1, 1), (1, 2)],  # unigrams vs unigrams+bigrams
    "tfidf__max_features": [30_000, 50_000],  # vocabulary size
    "tfidf__sublinear_tf": [True, False],  # log-scaled TF vs raw TF
    # Logistic Regression
    "clf__C": [0.1, 1.0, 10.0, 15.0],  # regularisation strength
    "clf__solver": ["lbfgs", "saga"],  # optimiser
}

Best parameters found:
  clf__C: 15.0
  clf__solver: saga
  tfidf__max_features: 50000
  tfidf__ngram_range: (1, 2)
  tfidf__sublinear_tf: True
Best CV F1 (weighted): 0.8517

=== Final Model (best params) ===
Accuracy : 0.8866
F1 (weighted): 0.8865
"""
