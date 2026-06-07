from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from preprocess import clean_tweet, load_data, preprocess_data
from cnn import build_cnn_model
from transformer_cnn import build_transformer_cnn_model, prepare_hybrid_data

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


def train_and_evaluate_cnn(X_train, y_train, X_test, y_test, examples):
    """Trains and evaluates the CNN model from src/cnn.py."""
    print("=== CNN Model ===")

    # 1. Label Encoding
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    num_classes = len(le.classes_)

    # 2. Tokenization & Padding
    max_words = 50_000
    max_len = 100
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding="post")
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding="post")

    vocab_size = min(len(tokenizer.word_index) + 1, max_words)

    # 3. Build Model
    model = build_cnn_model(vocab_size, num_classes, max_len)

    # 4. Train
    print(f"Training CNN for 3 epochs...")
    model.fit(
        X_train_pad,
        y_train_enc,
        epochs=3,
        batch_size=64,
        validation_data=(X_test_pad, y_test_enc),
        verbose=1,
    )

    # 5. Evaluate
    y_pred_probs = model.predict(X_test_pad)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\nCNN Performance:")
    print(f"Accuracy      : {accuracy_score(y_test_enc, y_pred):.4f}")
    print("Per-class report:")
    print(classification_report(y_test_enc, y_pred, target_names=le.classes_))

    # 6. Example Predictions
    print("=== Example Predictions (CNN) ===")
    cleaned_examples = [clean_tweet(t) for t in examples]
    ex_seq = tokenizer.texts_to_sequences(cleaned_examples)
    ex_pad = pad_sequences(ex_seq, maxlen=max_len, padding="post")

    ex_probs = model.predict(ex_pad)
    ex_preds = np.argmax(ex_probs, axis=1)

    for tweet, pred_idx, probs in zip(examples, ex_preds, ex_probs):
        pred_label = le.classes_[pred_idx]
        prob_str = "  ".join(f"{cls}={p:.2f}" for cls, p in zip(le.classes_, probs))
        print(f"\nTweet      : {tweet}")
        print(f"Prediction : {pred_label}")
        print(f"Confidence : {prob_str}")
    print("-" * 40 + "\n")


def train_and_evaluate_transformer_cnn(X_train, y_train, X_test, y_test, examples):
    print("=== Hybrid Windows Transformer-CNN Model ===")

    # 1. Target Label Encoding
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    num_classes = len(le.classes_)

    # 2. Tokenize using our safe hybrid function
    max_len = 100
    print("Mapping raw text safely on Windows environment...")
    X_train_trans = prepare_hybrid_data(X_train, max_length=max_len)
    X_test_trans = prepare_hybrid_data(X_test, max_length=max_len)

    # 3. Build Model Graph
    model = build_transformer_cnn_model(preset_name="bert_tiny_en_uncased", max_length=max_len, num_classes=num_classes)
    model.summary()

    # 4. Train Model Natively
    print("Training Hybrid Model...")
    model.fit(
        X_train_trans,
        y_train_enc,
        epochs=3,
        batch_size=64,
        validation_data=(X_test_trans, y_test_enc),
        verbose=1
    )

    # 5. Evaluate System Performance
    y_pred_probs = model.predict(X_test_trans)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\nTransformer-CNN Performance:")
    print(f"Accuracy: {accuracy_score(y_test_enc, y_pred):.4f}")
    print(classification_report(y_test_enc, y_pred, target_names=le.classes_))


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

    # CNN
    train_and_evaluate_cnn(X_train, y_train, X_test, y_test, examples)

    # CNN + Transformer
    train_and_evaluate_transformer_cnn(X_train, y_train, X_test, y_test, examples)


if __name__ == "__main__":
    main()
