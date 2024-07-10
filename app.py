# main.py
import config

# install some modules:

# pip install requests pandas
# pip install pandas nltk
# pip install pandas scikit-learn

import requests
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Replace 'YOUR_API_KEY' and 'YOUR_APP_ID' with your actual Edamam API key and ID
API_KEY = 'b126f2c98554725593aedca444b4cfe0'
APP_ID = '69e9ae01'
BASE_URL = 'https://api.edamam.com/search'

# Fetch recipes from the Edamam API with nutritional information and instructions URL
def fetch_recipes(query, app_id, api_key, number=100):
    params = {
        'q': query,
        'app_id': app_id,
        'app_key': api_key,
        'to': number,
        'nutrition-type': 'logging'  # Request for nutritional information
    }
    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()
    return response.json()

# Save recipe data including nutritional information to a CSV file
def save_recipes_to_csv(recipes, filename='recipes.csv'):
    recipe_list = []
    for recipe in recipes['hits']:
        recipe_info = {
            'title': recipe['recipe']['label'],
            'ingredients': ', '.join(recipe['recipe']['ingredientLines']),
            'quantities': ', '.join([f"{ingredient['quantity']} {ingredient['measure']} {ingredient['food']}" if 'quantity' in ingredient else '' for ingredient in recipe['recipe']['ingredients']]),
            'instructions_url': recipe['recipe']['url'],  # URL to instructions
            'calories': recipe['recipe']['calories'],  # Calories per serving
            'cuisine': ', '.join(recipe['recipe'].get('cuisineType', [])),
            'dietary_restrictions': ', '.join(recipe['recipe'].get('dietLabels', []))
        }
        recipe_list.append(recipe_info)

    df = pd.DataFrame(recipe_list)
    df.to_csv(filename, index=False)

# Load recipe data from a CSV file
def load_data(filename):
    return pd.read_csv(filename)

# Handle missing values in the recipe data
def handle_missing_values(df):
    df['ingredients'].fillna('', inplace=True)
    df['instructions_url'].fillna('', inplace=True)
    df['dietary_restrictions'].fillna('None', inplace=True)
    df['quantities'].fillna('', inplace=True)
    return df

# Normalize ingredient names by removing stopwords and lemmatizing
def normalize_ingredient_names(ingredients):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    normalized_ingredients = []
    for ingredient in ingredients.split(','):
        words = ingredient.lower().strip().split()
        filtered_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        normalized_ingredients.append(' '.join(filtered_words))
    
    return ', '.join(normalized_ingredients)

# Preprocess ingredients in the recipe data
def preprocess_ingredients(df):
    df['ingredients'] = df['ingredients'].apply(normalize_ingredient_names)
    return df

# Categorize recipes based on dietary restrictions
def categorize_recipes(df):
    df['category'] = df['dietary_restrictions'].apply(lambda x: 'Vegetarian' if 'vegetarian' in x.lower() else 'Non-Vegetarian')
    return df

# Save cleaned recipe data to a CSV file
def save_cleaned_data(df, filename='cleaned_recipes.csv'):
    df.to_csv(filename, index=False)

# Load cleaned data
def load_cleaned_data(filename='cleaned_recipes.csv'):
    df = pd.read_csv(filename)
    df['ingredients'] = df['ingredients'].astype(str)
    return df

# Recommend recipes based on ingredients
def recommend_recipes_by_ingredients(ingredients, df, top_n=5):
    vectorizer = CountVectorizer()
    ingredient_matrix = vectorizer.fit_transform(df['ingredients'])

    user_ingredients = ' '.join(ingredients)
    user_vector = vectorizer.transform([user_ingredients])
    
    similarity_scores = cosine_similarity(user_vector, ingredient_matrix)
    df['similarity'] = similarity_scores[0]
    
    top_recipes = df.sort_values(by='similarity', ascending=False).head(top_n)
    return top_recipes

# Recommend similar recipes based on user's liked recipes
def recommend_similar_recipes(liked_recipes, df, top_n=5):
    vectorizer = CountVectorizer()
    ingredient_matrix = vectorizer.fit_transform(df['ingredients'])
    
    liked_ingredients = ' '.join(df[df['title'].isin(liked_recipes)]['ingredients'])
    liked_vector = vectorizer.transform([liked_ingredients])
    
    similarity_scores = cosine_similarity(liked_vector, ingredient_matrix)
    df['similarity'] = similarity_scores[0]
    
    top_recipes = df.sort_values(by='similarity', ascending=False).head(top_n)
    return top_recipes

if __name__ == '__main__':
    # Download necessary NLTK data
    nltk.download('stopwords')
    nltk.download('wordnet')

    # Fetch and save recipes with extended information
    cuisine = 'Indian'
    app_id = APP_ID
    api_key = API_KEY
    number_of_recipes = 100  # You can adjust this number as needed

    recipes = fetch_recipes(cuisine, app_id, api_key, number=number_of_recipes)
    save_recipes_to_csv(recipes, filename='indian_recipes.csv')
    print('Recipes saved to indian_recipes.csv')

    # Load, preprocess, and save cleaned data with extended information
    df = load_data('indian_recipes.csv')
    df = handle_missing_values(df)
    df = preprocess_ingredients(df)
    df = categorize_recipes(df)
    save_cleaned_data(df, filename='cleaned_recipes.csv')
    print('Cleaned data saved to cleaned_recipes.csv')

    # Load cleaned data
    df = load_cleaned_data()

    # User inputs ingredients
    user_ingredients = input("Enter ingredients you have (comma separated): ").split(',')

    # Recommend recipes based on ingredients
    recommended_recipes = recommend_recipes_by_ingredients(user_ingredients, df)
    print("\nRecipes based on your ingredients:")
    for idx, row in recommended_recipes.iterrows():
        print(f"\nTitle: {row['title']}")
        print("Ingredients:")
        for ingredient, quantity in zip(row['ingredients'].split(','), row['quantities'].split(',')):
            print(f" - {quantity.strip()} {ingredient.strip()}")
        print(f"Calories per serving: {round(row['calories'],2)} Cal")
        print(f"Instructions URL: {row['instructions_url']}")

    # Ask if the user wants similar recipes based on their liked recipes
    liked_recipe_titles = input("\nEnter titles of recipes you like (comma separated): ").split(',')

    # Recommend similar recipes
    similar_recipes = recommend_similar_recipes(liked_recipe_titles, df)
    print("\nRecipes similar to those you like:")
    for idx, row in similar_recipes.iterrows():
        print(f"\nTitle: {row['title']}")
        print("Ingredients:")
        for ingredient, quantity in zip(row['ingredients'].split(','), row['quantities'].split(',')):
            print(f" - {quantity.strip()} {ingredient.strip()}")
        print(f"Calories per serving: {row['calories']}")
        print(f"Instructions URL: {row['instructions_url']}")