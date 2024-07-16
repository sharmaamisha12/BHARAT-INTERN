import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Step 1: Load the datasets
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# Step 2: Merge datasets and preprocess
data = movies.merge(credits, left_on='id', right_on='movie_id')
data['genres'] = data['genres'].apply(ast.literal_eval)
data['genres'] = data['genres'].apply(lambda x: [d['name'] for d in x])
data = data[['id', 'original_title', 'genres', 'vote_average', 'vote_count']]

# Step 3: Simulate User Ratings
# Here we generate simulated ratings based on vote_average and vote_count
data['rating'] = (data['vote_average'] * 2)  # Scale to a 1-5 scale
data['rating'] = data['rating'].clip(1, 5)  # Ensure ratings are between 1 and 5

# Create a dummy user rating some movies
ratings = pd.DataFrame({
    'user_id': np.random.choice([1, 2, 3], size=100),  # Simulate user IDs
    'movie_id': np.random.choice(data['id'], size=100),  # Simulate movie IDs
    'rating': np.random.randint(1, 6, size=100)  # Random ratings between 1 and 5
})

# Step 4: Create user-item interaction matrix
user_item_matrix = pd.pivot_table(ratings, index='user_id', columns='movie_id', values='rating')

# Step 5: Implement collaborative filtering
reader = Reader(rating_scale=(1, 5))
data_surprise = Dataset.load_from_df(ratings[['user_id', 'movie_id', 'rating']], reader)

trainset, testset = train_test_split(data_surprise, test_size=0.2)
model = SVD()
model.fit(trainset)

# Make predictions
predictions = model.test(testset)

# Evaluate the model
print("RMSE:", accuracy.rmse(predictions))

# Step 6: Generate movie recommendations
def get_recommendations(user_id, model, movies, n_recommendations=10):
    all_movie_ids = movies['id'].unique()
    predictions = [model.predict(user_id, movie_id) for movie_id in all_movie_ids]
    
    # Sort predictions by estimated rating
    top_movies = sorted(predictions, key=lambda x: x.est, reverse=True)
    
    # Get recommended movie IDs
    recommended_ids = [x[1] for x in top_movies[:n_recommendations]]
    recommended_movies = movies[movies['id'].isin(recommended_ids)]
    
    return recommended_movies[['original_title', 'vote_average']]

# Example usage
user_id = 1  # Replace with actual user ID
recommended_movies = get_recommendations(user_id, model, data)
print(recommended_movies)

# Step 7: Plotting
def plot_movie_recommendations(recommended_movies):
    plt.figure(figsize=(10, 6))
    plt.barh(recommended_movies['original_title'], recommended_movies['vote_average'], color='skyblue')
    plt.xlabel('Average Vote')
    plt.title('Top Movie Recommendations')
    plt.gca().invert_yaxis()  # Invert y axis to have the highest vote on top
    plt.show()

# Plot the recommended movies
plot_movie_recommendations(recommended_movies)
