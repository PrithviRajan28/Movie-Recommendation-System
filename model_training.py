import pandas as pd
from sklearn.decomposition import TruncatedSVD
import pickle

# Load data (MovieLens dataset as an example)
ratings = pd.read_csv('ratings.csv')  # Columns: userId, movieId, rating
movies = pd.read_csv('movies.csv')    # Columns: movieId, title

# Create a user-item matrix
user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Apply Matrix Factorization (SVD)
svd = TruncatedSVD(n_components=5)
user_movie_matrix_reduced = svd.fit_transform(user_movie_matrix)

# Save the model (reduced matrix)
with open('recommendation_model.pkl', 'wb') as model_file:
    pickle.dump(user_movie_matrix_reduced, model_file)
