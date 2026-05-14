from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline

from preprocess import clean_tweet, load_data, preprocess_data

# ── 2. Model Definitions ──────────────────────────────────────────────────────


def get_logistic_regression_pipeline():
    """Returns a Pipeline with TfidfVectorizer and LogisticRegression."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=50_000,
                    sublinear_tf=True,
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


def get_naive_bayes_pipeline():
    """Returns a Pipeline with TfidfVectorizer and MultinomialNB."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=50_000,
                    sublinear_tf=True,
                    min_df=2,
                ),
            ),
            ("clf", MultinomialNB()),
        ]
    )


# ── 3. Execution Logic ────────────────────────────────────────────────────────


def train_and_evaluate(pipeline, X_train, y_train, X_test, y_test, model_name):
    """Trains the given pipeline and prints evaluation metrics."""
    print(f"=== {model_name} ===")
    print(f"Training {model_name}...")
    pipeline.fit(X_train, y_train)
    print("Done.\n")

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"Accuracy      : {acc:.4f}")
    print(f"F1 (weighted) : {f1:.4f}\n")
    print("Per-class report:")
    print(classification_report(y_test, y_pred))
    print("-" * 40 + "\n")
    return pipeline


def run_examples(pipeline, examples, model_name):
    """Prints predictions for a list of example tweets."""
    print(f"=== Example Predictions ({model_name}) ===")
    cleaned_examples = [clean_tweet(t) for t in examples]
    predictions = pipeline.predict(cleaned_examples)
    probabilities = pipeline.predict_proba(cleaned_examples)
    classes = pipeline.classes_

    for tweet, pred, probs in zip(examples, predictions, probabilities):
        prob_str = "  ".join(f"{cls}={p:.2f}" for cls, p in zip(classes, probs))
        print(f"\nTweet      : {tweet}")
        print(f"Prediction : {pred}")
        print(f"Confidence : {prob_str}")
    print("\n" + "=" * 40 + "\n")


# ── 4. Main ───────────────────────────────────────────────────────────────────


def main():
    # Paths
    train_csv = "data/archive/twitter_training.csv"
    val_csv = "data/archive/twitter_validation.csv"

    # Data
    df_train, df_val = load_data(train_csv, val_csv)
    X_train, y_train, X_test, y_test = preprocess_data(df_train, df_val)

    examples = [
        "ive never seen a more well-put together clothing line",
        "this game is gorgeously horrendous",
        "Won meaningless games down the stretch the one year they had an opportunity to lock in a top-5 pick. Now they're picking 7th. The Sacramento Kings franchise will forever be bad due to their own incompetence.",
        "Wemby is a generational talent.  He is going to be a top player of all time. He is going to become more and more aggressive, and get defended more and more physically. He, like Shaq before him, is going to need to learn to help referees, to protect himself, and to be his own enforcer. This elbow, though. This ain't it.",
        "I need a room full of mirrors so I can be surrounded by winners.",
    ]

    # Logistic Regression
    lr_pipe = get_logistic_regression_pipeline()
    train_and_evaluate(lr_pipe, X_train, y_train, X_test, y_test, "Logistic Regression")
    run_examples(lr_pipe, examples, "Logistic Regression")

    # Multinomial Naive Bayes
    nb_pipe = get_naive_bayes_pipeline()
    train_and_evaluate(
        nb_pipe, X_train, y_train, X_test, y_test, "Multinomial Naive Bayes"
    )
    run_examples(nb_pipe, examples, "Multinomial Naive Bayes")


if __name__ == "__main__":
    main()
