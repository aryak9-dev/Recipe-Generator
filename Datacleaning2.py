import pandas as pd
from sklearn.model_selection import train_test_split
import json

# Load cleaned and annotated data
file_path = "/Users/arya/code/recipe_gen/archive/processed_recipes.json"
with open(file_path, "r") as f:
    data = json.load(f)

# Convert JSON to DataFrame for easy processing
data_df = pd.DataFrame(data)
print("Data Loaded:", data_df.head())

# Features (inputs) and targets (outputs)
features = data_df[['ingredients', 'instructions']]  # Input features
targets = data_df['recipe_name']  # Output/target

# Train-Test Split
X_train, X_temp, y_train, y_temp = train_test_split(features, targets, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Training Data: {X_train.shape}")
print(f"Validation Data: {X_val.shape}")
print(f"Test Data: {X_test.shape}")

# Function to structure prompt-response pairs
def create_prompt_response(features, targets):
    data_pairs = []
    for i in range(len(features)):
        prompt = f"Create a recipe using {', '.join(features.iloc[i]['ingredients'])}."
        response = {
            "recipe_name": targets.iloc[i],
            "ingredients": features.iloc[i]['ingredients'],
            "instructions": features.iloc[i]['instructions']
        }
        data_pairs.append({"prompt": prompt, "response": response})
    return data_pairs

# Create prompt-response pairs
train_data = create_prompt_response(X_train, y_train)
val_data = create_prompt_response(X_val, y_val)
test_data = create_prompt_response(X_test, y_test)

# Save to JSON
output_folder = "/Users/arya/code/recipe_gen/archive/"
with open(output_folder + "train_data.json", "w") as f:
    json.dump(train_data, f, indent=2)
with open(output_folder + "val_data.json", "w") as f:
    json.dump(val_data, f, indent=2)
with open(output_folder + "test_data.json", "w") as f:
    json.dump(test_data, f, indent=2)

print("Prompt-response data saved for training, validation, and testing!")
