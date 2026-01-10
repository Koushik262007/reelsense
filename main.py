import pandas as pd
import ast

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# Step 1: Load datasets
# =========================
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

# =========================
# Step 2: Merge datasets
# =========================
movies = movies.merge(credits, on="title")

# Keep only useful columns
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

print("After merge:", movies.shape)

# =========================
# Step 3: Remove missing values
# =========================
movies.dropna(inplace=True)
print("After dropna:", movies.shape)

# =========================
# Step 4: Convert genres & keywords
# =========================
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

# =========================
# Step 5: Convert cast (top 3 actors)
# =========================
def convert_cast(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

movies['cast'] = movies['cast'].apply(convert_cast)

# =========================
# Step 6: Extract director from crew
# =========================
def fetch_director(text):
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            return [i['name']]
    return []

movies['crew'] = movies['crew'].apply(fetch_director)

# =========================
# Step 7: Clean overview text
# =========================
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# =========================
# Step 8: Create TAGS column
# =========================
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
# Step 9: Vectorization using TF-IDF
# =========================
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')

vectors = tfidf.fit_transform(new_df['tags']).toarray()

print("Vector shape:", vectors.shape)

# =========================
# Step 10: Cosine similarity
# =========================
similarity = cosine_similarity(vectors)

print("Similarity shape:", similarity.shape)
# =========================
# Step 11: Recommendation function
# =========================
def recommend(movie):
    movie = movie.lower()

    if movie not in new_df['title'].str.lower().values:
        print("Movie not found in database")
        return

    index = new_df[new_df['title'].str.lower() == movie].index[0]

    distances = similarity[index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    print(f"\nRecommended movies for '{movie.title()}':\n")

    for i in movie_list:
        print(new_df.iloc[i[0]].title)

recommend("Avatar")


# =========================
# Final check
# =========================
print(new_df.head())
