import pickle
import streamlit as st
import numpy as np

st.header('Book Recommender System Using Machine Learning')
model = pickle.load(open('./nn_model.pkl', 'rb'))
book_names = pickle.load(open('./nn_book_names.pkl', 'rb'))
final_rating = pickle.load(open('./nn_final_rating.pkl', 'rb'))
book_pivot = pickle.load(open('./nn_book_pivot.pkl', 'rb'))

def fetch_book_info(book_name):
    info_list = []
    for name in book_name:
        idx = np.where(final_rating['title'] == name)[0][0]
        author = final_rating.iloc[idx]['author']
        year = final_rating.iloc[idx]['year']
        publisher = final_rating.iloc[idx]['publisher']
        num_of_ratings = final_rating.iloc[idx]['num_of_rating']
        info_list.append((author, year, publisher, num_of_ratings))
    return info_list

def fetch_poster(suggestion):
    book_name = [book_pivot.index[book_id] for book_id in suggestion]
    ids_index = [np.where(final_rating['title'] == name)[0][0] for name in book_name[0]]
    poster_url = [final_rating.iloc[idx]['image_url'] for idx in ids_index]
    return poster_url

def recommend_book(book_name):
    books_list = []
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)
    poster_url = fetch_poster(suggestion)
    for i in range(len(suggestion)):
        books = book_pivot.index[suggestion[i]]
        for j in books:
            books_list.append(j)
    return books_list, poster_url

selected_books = st.selectbox(
    "Type or select a book from the dropdown",
    book_names
)

if st.button('Show Recommendation'):
    recommended_books, poster_url = recommend_book(selected_books)

    for i in range(len(recommended_books)):
        col1, col2 = st.columns([1, 4])

        with col1:
            st.image(poster_url[i], caption='', use_column_width=True)
        with col2:
            book_info = fetch_book_info([recommended_books[i]])[0]
            st.markdown(
                f"""
                <div style="font-size: 18px; font-weight: bold; margin-bottom: 10px;">{recommended_books[i]}</div>
                <div style="margin-bottom: 10px;">Author: {book_info[0]}</div>
                <div style="margin-bottom: 10px;">Year: {book_info[1]}</div>
                <div style="margin-bottom: 10px;">Publisher: {book_info[2]}</div>
                <div style="margin-bottom: 10px;">Number of Ratings: {book_info[3]}</div>
                """,
                unsafe_allow_html=True
            )

#python -m streamlit run app.py 


























            
