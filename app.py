from flask import Flask, request, render_template
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the pre-trained recommendation model and data
model = pickle.load(open('recommendation_model.pkl', 'rb'))
movie_titles = pd.read_csv('movies.csv')  # Assuming you have a file with movie titles and IDs

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = int(request.form['user_id'])
    
    # Get recommendations for the user
    user_ratings = model[user_id]  # Assuming the model provides ratings for the user
    similarity_matrix = cosine_similarity(user_ratings.reshape(1, -1), model).flatten()
    
    # Get top 5 recommendations
    recommended_movies_indices = similarity_matrix.argsort()[-6:-1][::-1]
    recommendations = movie_titles.iloc[recommended_movies_indices]

    return render_template('recommendations.html', movies=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
