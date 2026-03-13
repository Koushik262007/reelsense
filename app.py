import pandas as pd
import ast
import streamlit as st
import requests

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# TMDB API KEY (from secrets)
# =========================
TMDB_API_KEY = st.secrets["3313c5a7de33d64379168a116dcc80c3"]
PLACEHOLDER_POSTER = "https://via.placeholder.com/500x750?text=No+Poster"

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0B132B;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #EAEAEA;
    }
    p, span, label {
        color: #C7C7C7;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# Sidebar
# =========================
st.sidebar.title("Movie Recommender")
st.sidebar.markdown("Get movie suggestions based on content similarity.")
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **How it works**
    - Movie metadata (overview, genres, cast, director)
    - TF-IDF vectorization
    - Cosine similarity
    """
)
st.sidebar.markdown("---")
st.sidebar.info("Built by Koushik")

# =========================
# Fetch poster using TMDB API
# =========================
@st.cache_data(show_spinner=False)
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{int(movie_id)}"
    params = {
        "api_key": TMDB_API_KEY,
        "language": "en-US"
    }

    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()

        poster_path = data.get("poster_path")
        if poster_path:
            return "https://image.tmdb.org/t/p/w500" + poster_path

    except requests.exceptions.RequestException:
        # Any network / API error
        return None

    return None


# =========================
# Load datasets
# =========================
@st.cache_data
def load_data():
    movies_df = pd.read_csv("tmdb_5000_movies.csv")
    credits_df = pd.read_csv("tmdb_5000_credits.csv")

    movies_df = movies_df.merge(credits_df, on="title")
    movies_df = movies_df[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    movies_df.dropna(inplace=True)

    return movies_df


movies = load_data()

# =========================
# Data preprocessing
# =========================
def convert(text):
    return [i['name'] for i in ast.literal_eval(text)]

def convert_cast(text):
    return [i['name'] for i in ast.literal_eval(text)[:3]]

def fetch_director(text):
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            return [i['name']]
    return []

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert_cast)
movies['crew'] = movies['crew'].apply(fetch_director)
movies['overview'] = movies['overview'].apply(lambda x: x.split())

movies['tags'] = (
    movies['overview']
    + movies['genres']
    + movies['keywords']
    + movies['cast']
    + movies['crew']
)

new_df = movies[['movie_id', 'title', 'tags']].copy()
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

# =========================
# Vectorization
# =========================
@st.cache_data
def compute_similarity(tags):
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    vectors = tfidf.fit_transform(tags).toarray()
    return cosine_similarity(vectors)

similarity = compute_similarity(new_df['tags'])


# =========================
# Recommendation function
# =========================
def recommend(movie):
    movie = movie.lower()

    if movie not in new_df['title'].str.lower().values:
        return []

    index = new_df[new_df['title'].str.lower() == movie].index[0]
    distances = similarity[index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommendations = []
    for i in movie_list:
        movie_id = new_df.iloc[i[0]].movie_id
        title = new_df.iloc[i[0]].title
        poster = fetch_poster(movie_id)
        recommendations.append((title, poster))

    return recommendations

# =========================
# Streamlit UI
# =========================
st.image("assets/logo.png", width=90)

st.markdown(
    "<h1 style='text-align: center;'>Movie Recommendation System</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h1 style='text-align: center;'>ReelSense</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; color:#9CA3AF;'>Smart movie recommendations powered by machine learning</p>",
    unsafe_allow_html=True
)


st.markdown(
    "<p style='text-align: center;'>Select a movie to get similar recommendations</p>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; color:#9CA3AF;'>Content-based movie recommendation using machine learning</p>",
    unsafe_allow_html=True
)
search_query = st.text_input(
    "🔍 Search for a movie",
    placeholder="Type movie name (e.g. Avatar)"
)

# Find matching movies
matched_movies = new_df[new_df['title'].str.contains(
    search_query, case=False, na=False
)]

# Show suggestions only if user types
if search_query:
    selected_movie = st.selectbox(
        " Select from results",
        matched_movies['title'].values
    )
else:
    selected_movie = None

if st.button(" Recommend Movies"):
    with st.spinner("Finding best movies for you... "):
        recommendations = recommend(selected_movie)

    if not recommendations:
        st.warning("No recommendations found.")
    else:
        st.subheader("✨ Recommended Movies For You")
        cols = st.columns(len(recommendations))

        for col, (title, poster) in zip(cols, recommendations):
            with col:
                st.image(
                    poster if poster else PLACEHOLDER_POSTER,
                    use_container_width=True
                )
                st.markdown(f"**🎬 {title}**")

