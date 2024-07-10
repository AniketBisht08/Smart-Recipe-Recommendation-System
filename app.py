# main.py
import config

# Data collection

import requests
import pandas as pd

# Replace 'YOUR_API_KEY' with your actual Spoonacular API key
API_KEY = config.YOUR_API_KEY
BASE_URL = 'https://api.spoonacular.com/recipes/complexSearch'

def fetch_recipes(cuisine, api_key, number=100, offset=0):
    params = {
        'cuisine': cuisine,
        'apiKey': api_key,
        'number': number,
        'offset': offset,
        'addRecipeInformation': True
    }
    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()
    return response.json()

def save_recipes_to_csv(recipes, filename='recipes.csv'):
    recipe_list = []
    for recipe in recipes['results']:
        recipe_info = {
            'id': recipe['id'],
            'title': recipe['title'],
            'ingredients': ', '.join([ingredient['name'] for ingredient in recipe.get('extendedIngredients', [])]),
            'instructions': recipe.get('instructions', ''),
            'cuisine': cuisine,
            'dietary_restrictions': ', '.join(recipe.get('diets', []))
        }
        recipe_list.append(recipe_info)

    df = pd.DataFrame(recipe_list)
    df.to_csv(filename, index=False)

if __name__ == '__main__':
    cuisine = 'Indian'
    api_key = API_KEY
    number_of_recipes = 1000  # You can adjust this number as needed
    offset = 0  # Use this to paginate through results if needed

    recipes = fetch_recipes(cuisine, api_key, number=number_of_recipes, offset=offset)
    save_recipes_to_csv(recipes, filename='indian_recipes.csv')
    print('Recipes saved to indian_recipes.csv')




# Data cleaning and preprocessing

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(filename):
    return pd.read_csv(filename)

def handle_missing_values(df):
    # Fill missing ingredients with an empty string
    df['ingredients'].fillna('', inplace=True)
    # Fill missing instructions with an empty string
    df['instructions'].fillna('', inplace=True)
    # Fill missing dietary restrictions with 'None'
    df['dietary_restrictions'].fillna('None', inplace=True)
    return df

def normalize_ingredient_names(ingredients):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    normalized_ingredients = []
    for ingredient in ingredients.split(','):
        words = ingredient.lower().strip().split()
        filtered_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        normalized_ingredients.append(' '.join(filtered_words))
    
    return ', '.join(normalized_ingredients)

def preprocess_ingredients(df):
    df['ingredients'] = df['ingredients'].apply(normalize_ingredient_names)
    return df

def categorize_recipes(df):
    # Example categorization by dietary restrictions
    df['category'] = df['dietary_restrictions'].apply(lambda x: 'Vegetarian' if 'vegetarian' in x.lower() else 'Non-Vegetarian')
    return df

def save_cleaned_data(df, filename='cleaned_recipes.csv'):
    df.to_csv(filename, index=False)

if __name__ == '__main__':
    filename = 'indian_recipes.csv'
    df = load_data(filename)

    # Handle missing values
    df = handle_missing_values(df)

    # Normalize ingredient names
    df = preprocess_ingredients(df)

    # Categorize recipes
    df = categorize_recipes(df)

    # Save cleaned data
    save_cleaned_data(df, filename='cleaned_recipes.csv')
    print('Cleaned data saved to cleaned_recipes.csv')

    
# Building model

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load cleaned data
def load_cleaned_data(filename='cleaned_recipes.csv'):
    df = pd.read_csv(filename)
    # Ensure all ingredients are strings
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
    # Load cleaned data
    df = load_cleaned_data()

    # User inputs ingredients
    user_ingredients = input("Enter ingredients you have (comma separated): ").split(',')

    # Recommend recipes based on ingredients
    recommended_recipes = recommend_recipes_by_ingredients(user_ingredients, df)
    print("\nRecipes based on your ingredients:")
    print(recommended_recipes[['title', 'ingredients', 'instructions']])

    # Ask if the user wants similar recipes based on their liked recipes
    liked_recipe_titles = input("\nEnter titles of recipes you like (comma separated): ").split(',')

    # Recommend similar recipes
    similar_recipes = recommend_similar_recipes(liked_recipe_titles, df)
    print("\nRecipes similar to those you like:")
    print(similar_recipes[['title', 'ingredients', 'instructions']])