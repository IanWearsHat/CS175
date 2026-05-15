from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    Conv1D,
    GlobalMaxPooling1D,
    Dense,
    Dropout,
)


def build_cnn_model(vocab_size, num_classes, max_length, embedding_dim=128):
    """
    Builds and compiles a 1D CNN for text classification.
    """
    model = Sequential(
        [
            # Embedding layer: maps words to dense vectors
            Embedding(vocab_size, embedding_dim, input_length=max_length),
            # Conv1D: extracts local features (ngrams)
            Conv1D(128, 5, activation="relu"),
            # Global Max Pooling: takes the most important feature from each filter
            GlobalMaxPooling1D(),
            # Dense layers for classification
            Dense(64, activation="relu"),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    return model
