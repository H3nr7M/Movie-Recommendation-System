import pandas as pd
import streamlit as st
import difflib #avoid miswriting names
from sklearn.feature_extraction.text import TfidfVectorizer #frecuencia de palabras
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image


st.image('log.jpg', use_column_width=True)

movies_data = pd.read_csv('movies.csv')

selected_features = ['genres','keywords','tagline','cast','director']

# replacing the null valuess with null string
for feature in selected_features:
  movies_data[feature] = movies_data[feature].fillna('')

# combining all the 5 selected features
combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']

# converting the text data to feature vectors
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
# getting the similarity scores using cosine similarity
similarity = cosine_similarity(feature_vectors)

list_of_all_titles = movies_data['title'].tolist()

with st.container():
    st.write("Welcome to the Movie Recommendation System!")

    with st.form(key='movie_search_form'):
        movie_name = st.text_input('Enter your favorite movie name:')
        submit_button = st.form_submit_button(label='Find Similar Movies')

        if movie_name or submit_button:
            find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
            if find_close_match:
                close_match = find_close_match[0]
                index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
                similarity_score = list(enumerate(similarity[index_of_the_movie]))
                sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

                st.header('Movies suggested for you:')
                for i, movie in enumerate(sorted_similar_movies):
                    index = movie[0]
                    title_from_index = movies_data[movies_data.index == index]['title'].values[0]
                    st.write(f"{i+1}. {title_from_index}")
                    if i >= 4:
                        break
            else:
                st.warning("Please enter a valid movie name.")