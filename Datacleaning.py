import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = "/Users/arya/code/recipe_gen/archive/RecipeNLG_dataset.csv"  
data = pd.read_csv(file_path)

# Understand the data
print(data.info())  # Check data types and null values

# Clean the data
data = data.dropna(subset=['ingredients', 'directions'])  # Drop nulls in critical columns
data['steps'] = data['directions'].apply(lambda x: x.split(". "))  # Split directions into steps

# Normalize ingredients
ingredient_map = {
    "granulated sugar": "sugar",
    "all-purpose flour": "flour"
}

def normalize_ingredients(ingredient_list):
    return [ingredient_map.get(ing.strip().lower(), ing.strip().lower()) for ing in ingredient_list if ing]

data['ingredients'] = data['ingredients'].apply(lambda x: normalize_ingredients(x.split(", ")))

# Structure recipes
def structure_recipe(row):
    return {
        "recipe_name": row['title'],
        "ingredients": row['ingredients'],
        "instructions": row['steps'],
        "metadata": {
            "source": row['source'],  # Metadata includes source
            "link": row['link']  # Link to the recipe
        }
    }
structured_data = data.apply(structure_recipe, axis=1)

# Save structured data to JSON
output_file = "/Users/arya/code/recipe_gen/archive/processed_recipes.json"
with open(output_file, "w") as f:
    json.dump(structured_data.tolist(), f, indent=2)

# Optional visualization: Recipe sources
if data['source'].nunique() > 1:  # Plot source distribution only if multiple sources exist
    sns.countplot(y=data['source'], order=data['source'].value_counts().index)
    plt.title("Recipe Source Distribution")
    plt.xlabel("Count")
    plt.ylabel("Source")
    plt.show()
else:
    print("Insufficient data for source distribution visualization.")

