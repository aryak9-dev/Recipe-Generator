import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow"])

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense

# Hyperparameters
embedding_dim = 256
lstm_units = 512
vocab_size = 8000  # Should match your tokenizer's vocab size
max_seq_length = 100  # Max tokens in prompt or response

def create_model(vocab_size, embedding_dim, lstm_units, max_seq_length):
    # Encoder
    encoder_input = Input(shape=(max_seq_length,), name="encoder_input")  # Fixed input shape
    encoder_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(encoder_input)  # Removed mask_zero
    encoder_lstm = LSTM(lstm_units, return_state=True, name="encoder_lstm")  # Removed masking dependency
    encoder_output, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_input = Input(shape=(max_seq_length,), name="decoder_input")  # Fixed input shape
    decoder_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(decoder_input)  # Removed mask_zero
    decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True, name="decoder_lstm")
    decoder_output, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = Dense(vocab_size, activation="softmax", name="output_layer")(decoder_output)

    # Seq2Seq Model
    model = Model([encoder_input, decoder_input], decoder_dense)
    
    # Compile the model
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model

# Create the model
model = create_model(vocab_size, embedding_dim, lstm_units, max_seq_length)

# Model summary
model.summary()
