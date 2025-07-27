import sentencepiece as spm
import pandas as pd
import json
import os

# Load training data
train_file = "/Users/arya/code/recipe_gen/archive/train_data.json"
with open(train_file, "r") as f:
    train_data = json.load(f)

# Prepare text data for SentencePiece
text_corpus = []
for pair in train_data:
    prompt = pair["prompt"]
    response = json.dumps(pair["response"])  # Convert response to a string
    text_corpus.extend([prompt, response])

# Specify the dataset folder and ensure it exists
dataset_folder = "/Users/arya/code/recipe_gen/archive/"
os.makedirs(dataset_folder, exist_ok=True)

# Save text data to the dataset folder for SentencePiece
corpus_file = os.path.join(dataset_folder, "recipe_corpus.txt")
with open(corpus_file, "w", encoding="utf-8") as f:
    for line in text_corpus:
        f.write(line + "\n")

print(f"Corpus saved at: {corpus_file}")

# Train SentencePiece tokenizer
spm.SentencePieceTrainer.train(
    input=corpus_file,
    model_prefix=os.path.join(dataset_folder, "recipe_tokenizer"),
    vocab_size=8000,
    model_type="bpe"  # Byte-Pair Encoding
)

# Load tokenizer
sp = spm.SentencePieceProcessor(model_file=os.path.join(dataset_folder, "recipe_tokenizer.model"))

# Example usage
example_prompt = "Create a recipe using tofu and spinach."
tokenized_prompt = sp.encode(example_prompt, out_type=str)
tokenized_prompt_readable = [token.replace('▁', '_') for token in tokenized_prompt]
print("Tokenized Prompt:", tokenized_prompt_readable)

# Decode tokens back to text
decoded_prompt = sp.decode(sp.encode(" ".join(tokenized_prompt_readable), out_type=int))
print("Decoded Prompt:", decoded_prompt)


