import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, GlobalMaxPooling1D, Dense, Dropout
import keras_hub
from transformers import AutoTokenizer
import numpy as np

def build_transformer_cnn_model(preset_name="bert_tiny_en_uncased", max_length=100, num_classes=3):
    """
    Builds an embedded Transformer-CNN model.
    Bypasses the Windows 'tensorflow-text' crash by mapping inputs via raw Keras layers.
    """
    # 1. Load the core mathematical backbone layer natively
    encoder_backbone = keras_hub.models.BertBackbone.from_preset(preset_name)
    encoder_backbone.trainable = False

    # 2. Re-create the standard architectural inputs explicitly
    input_ids = Input(shape=(max_length,), dtype=tf.int32, name="token_ids")
    padding_mask = Input(shape=(max_length,), dtype=tf.int32, name="padding_mask")
    segment_ids = Input(shape=(max_length,), dtype=tf.int32, name="segment_ids")

    # 3. Pass token tensors directly into the active graph
    transformer_outputs = encoder_backbone({
        "token_ids": input_ids,
        "padding_mask": padding_mask,
        "segment_ids": segment_ids
    })
    
    # Extract the sequence output matrix (Shape: batch_size, max_length, hidden_dim)
    sequence_output = transformer_outputs["sequence_output"]

    # 4. Local Feature Extraction (1D CNN)
    conv = Conv1D(filters=128, kernel_size=5, activation="relu", name="cnn_ngram_extractor")(sequence_output)
    pool = GlobalMaxPooling1D()(conv)

    # 5. Classification Head
    dense_hidden = Dense(64, activation="relu")(pool)
    dropout = Dropout(0.5)(dense_hidden)
    output = Dense(num_classes, activation="softmax", name="predictions")(dropout)

    # Assemble the functional Keras graph
    model = Model(inputs=[input_ids, padding_mask, segment_ids], outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def prepare_hybrid_data(texts, max_length=100):
    """
    Safely tokenizes text on Windows using Hugging Face's native tokenizer,
    formatting it into the exact structural dictionary expected by KerasHub.
    """
    # Use standard DistilBERT or BERT uncased tokenizer parameters
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    encodings = tokenizer(
        list(texts),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="np"  # Return NumPy arrays directly
    )
    
    # Map Hugging Face's naming conventions directly to KerasHub's input tokens
    return {
        "token_ids": encodings["input_ids"],
        "padding_mask": encodings["attention_mask"],
        "segment_ids": encodings.get("token_type_ids", np.zeros_like(encodings["input_ids"]))
    }