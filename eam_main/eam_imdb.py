import pandas as pd
import numpy as np
from imdb import IMDb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import MultiLabelBinarizer
import os
import time

# MMF class definition (directly embedded)
class MMF:
    def __init__(self, R, item_attrs, k=30, alpha=0.005, beta=0.01, epochs=500):
        self.R = R
        self.item_attrs = item_attrs
        self.num_users, self.num_items = R.shape
        self.num_genres = item_attrs.shape[1]
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.epochs = epochs

        self.U = np.random.normal(scale=1. / k, size=(self.num_users, k))
        self.F = np.random.normal(scale=1. / k, size=(self.num_genres, k))
        self.omega = np.random.normal(0.1, 0.01, (self.num_users, self.num_genres))

        self.theta = np.zeros_like(self.item_attrs, dtype=np.float32)
        non_zero_mask = self.item_attrs > 0
        self.theta[non_zero_mask] = np.random.normal(loc=0.1, scale=0.01, size=np.sum(non_zero_mask))

        self.user_indices, self.item_indices = np.where(R > 0)
        self.samples = list(zip(self.user_indices, self.item_indices))

    def predict(self, u, i):
        genres = np.where(self.item_attrs[i] > 0)[0]
        total = 0.0
        for g in genres:
            dot_val = np.dot(self.U[u], self.F[g])
            dot_val = np.clip(dot_val, -10, 10)
            total += self.omega[u, g] * self.theta[i, g] * dot_val
        return np.clip(total / (len(genres) + 1e-8), 1, 5)

    def train(self, verbose=True):
        for epoch in range(self.epochs):
            if epoch < 50:
                current_alpha = self.alpha
            else:
                current_alpha = self.alpha * (0.9 ** ((epoch - 50) // 10))

            np.random.shuffle(self.samples)

            for u, i in self.samples:
                r_ui = self.R[u, i]
                r_hat = self.predict(u, i)
                e_ui = r_ui - r_hat
                genres = np.where(self.item_attrs[i] > 0)[0]

                for g in genres:
                    dot = np.dot(self.U[u], self.F[g])
                    dot = np.clip(dot, -10, 10)

                    grad_U = e_ui * self.omega[u, g] * self.theta[i, g] * self.F[g]
                    grad_F = e_ui * self.omega[u, g] * self.theta[i, g] * self.U[u]
                    grad_omega = e_ui * self.theta[i, g] * dot
                    grad_theta = e_ui * self.omega[u, g] * dot

                    for grad in [grad_U, grad_F]:
                        norm = np.linalg.norm(grad)
                        if norm > 1.0:
                            grad /= norm

                    self.U[u] += current_alpha * (grad_U - self.beta * self.U[u])
                    self.F[g] += current_alpha * (grad_F - self.beta * self.F[g])

                    new_omega = self.omega[u, g] + current_alpha * (grad_omega - self.beta * self.omega[u, g])
                    new_theta = self.theta[i, g] + current_alpha * (grad_theta - self.beta * self.theta[i, g])
                    self.omega[u, g] = np.clip(new_omega, 0.1, 5.0)
                    self.theta[i, g] = np.clip(new_theta, 0.1, 5.0)

            if verbose and (epoch + 1) % 10 == 0:
                rmse = self.evaluate_rmse()
                print(f"Epoch {epoch + 1}/{self.epochs}, RMSE: {rmse:.4f}")

    def evaluate_rmse(self):
        mse = 0.0
        for u, i in self.samples:
            r_ui = self.R[u, i]
            r_hat = self.predict(u, i)
            mse += (r_ui - r_hat) ** 2
        return np.sqrt(mse / len(self.samples))


if __name__ == "__main__":
    print("Loading MovieLens movies...")
    movies = pd.read_csv('../data/ml-100k/u.item', sep='|', encoding='latin-1', header=None, usecols=[0, 1, 2],
                         names=['movie_id', 'title', 'release_date'])
    movies['year'] = movies['release_date'].str.extract(r'(\d{4})')
    movies['movie_id'] = movies['movie_id'].astype(int)

    print("Querying IMDb (this may take time)...")
    ia = IMDb()
    cached_file = "movies_with_imdb.csv"

    if os.path.exists(cached_file):
        movies = pd.read_csv(cached_file)
        movies['actors'] = movies['actors'].apply(eval)
        movies['directors'] = movies['directors'].apply(eval)
    else:
        directors_list, actors_list, plots = [], [], []
        for idx, row in movies.iterrows():
            title, year = row['title'], row['year']
            try:
                results = ia.search_movie(title)
                matched = None
                for res in results:
                    if 'year' in res and year and abs(int(res['year']) - int(year)) <= 1:
                        matched = ia.get_movie(res.movieID)
                        break
                if matched:
                    directors = [d['name'] for d in matched.get('director', [])]
                    actors = [a['name'] for a in matched.get('cast', [])[:5]]
                    plot = matched.get('plot', [''])[0].split('::')[0]
                else:
                    directors, actors, plot = [], [], ''
            except Exception as e:
                directors, actors, plot = [], [], ''
            directors_list.append(directors)
            actors_list.append(actors)
            plots.append(plot)
            time.sleep(0.5)

        movies['directors'] = directors_list
        movies['actors'] = actors_list
        movies['plot'] = plots
        movies.to_csv(cached_file, index=False)

    print("Running LDA on plot summaries...")
    plots = movies['plot'].fillna('').astype(str).tolist()
    vectorizer = CountVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(plots)
    lda = LatentDirichletAllocation(n_components=50, random_state=42)
    topic_matrix = lda.fit_transform(X)
    movies['topic_id'] = topic_matrix.argmax(axis=1)

    def build_attribute_matrix(movies, field_name):
        mlb = MultiLabelBinarizer()
        mat = mlb.fit_transform(movies[field_name])
        return mat, mlb.classes_

    print("Encoding attributes...")
    actor_mat, actor_list = build_attribute_matrix(movies, 'actors')
    director_mat, director_list = build_attribute_matrix(movies, 'directors')
    topic_mat = pd.get_dummies(movies['topic_id']).values

    genre_data = pd.read_csv('u.item', sep='|', encoding='latin-1', header=None, usecols=range(5, 24))
    genre_mat = genre_data.values

    print("Merging all attributes...")
    item_attrs_all = np.concatenate([genre_mat, actor_mat, director_mat, topic_mat], axis=1)
    np.save("item_attrs_all.npy", item_attrs_all)
    print("Attribute matrix shape:", item_attrs_all.shape)

    print("Loading rating matrix...")
    R = np.zeros((943, 1682))
    ratings = pd.read_csv('u.data', sep='\t', header=None, names=['user', 'item', 'rating', 'timestamp'])
    for row in ratings.itertuples():
        R[row.user - 1, row.item - 1] = row.rating

    print("Training MMF model...")
    mmf = MMF(R, item_attrs_all, k=30, alpha=0.01, beta=0.005, epochs=200)
    mmf.train()
