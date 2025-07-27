import base64
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "sentencepiece"])
import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import json
import sentencepiece as spm

# Hyperparameters (Make sure these match with your model)
embedding_dim = 256
lstm_units = 512
vocab_size = 8000
max_seq_length = 100

# Load SentencePiece Tokenizer
sp = spm.SentencePieceProcessor(model_file="/Users/arya/code/recipe_gen/archive/recipe_tokenizer.model")
print("sentence piece done")
# Load training data
train_file = "/Users/arya/code/recipe_gen/archive/train_data.json"
with open(train_file, "r") as f:
    train_data = json.load(f)
print("training data loaded")

# Prepare the data
encoder_input_data = []
decoder_input_data = []
decoder_target_data = []

for pair in train_data:
    prompt = pair["prompt"]
    response = pair["response"]
    
    # Tokenize the input (prompt) and output (response)
    tokenized_prompt = sp.encode(prompt, out_type=int)
    # print("data tokenized done")
    
   # Process and concatenate the parts of the response into a string
    recipe_name = response["recipe_name"]
    ingredients = " ".join(response["ingredients"])  # Concatenate ingredients into one string
    instructions = " ".join(response["instructions"])  # Concatenate instructions into one string
    # print("concatnation done")
    # Combine recipe name, ingredients, and instructions as a single string
    full_response = str(recipe_name) + " " + " ".join(map(str, ingredients)) + " " + " ".join(map(str, instructions))
    
    # Tokenize the full response
    tokenized_response = sp.encode(full_response, out_type=int)
    
    # Add the start and end tokens for the decoder input/output
    tokenized_response_input = [sp.PieceToId('<s>')] + tokenized_response
    tokenized_response_output = tokenized_response + [sp.PieceToId('</s>')]
    
    encoder_input_data.append(tokenized_prompt)
    decoder_input_data.append(tokenized_response_input)
    decoder_target_data.append(tokenized_response_output)
print("data training complete")
# Pad sequences to ensure they all have the same length
encoder_input_data = pad_sequences(encoder_input_data, maxlen=max_seq_length, padding='post')
decoder_input_data = pad_sequences(decoder_input_data, maxlen=max_seq_length, padding='post')
decoder_target_data = pad_sequences(decoder_target_data, maxlen=max_seq_length, padding='post')

# Convert target data to numpy array with the shape (batch_size, sequence_length, 1)
decoder_target_data = np.expand_dims(decoder_target_data, -1)

# Or create a new model
from projectt.model_structure1 import create_model
model = create_model(vocab_size, embedding_dim, lstm_units, max_seq_length)

# Train the model
model.fit(
    [encoder_input_data, decoder_input_data], 
    decoder_target_data, 
    batch_size=8, 
    epochs=30,  # You can adjust epochs as per your need
    validation_split=0.2
)
logging.basicConfig(level=logging.DEBUG)
# Save the trained model
model.save("/Users/arya/code/recipe_gen/archive/Trained_recipe_model.h5")

