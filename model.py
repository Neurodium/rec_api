import pandas as pd
import os
from scipy import spatial
import pickle

model_filename = "articles_embeddings_pca.pickle"

# API version
__version__ = "0.2.0"

articles_df = pd.read_csv('article.csv', index_col=0)

best_articles = pd.read_csv('best_articles.csv', index_col=0)

with open(model_filename, 'rb') as file:
    model = pickle.load(file)


def get_best_articles():
    dict_best_art = {}
    articles = best_articles['article_id'].sample(5)
    for article in articles:
        dict_best_art[article] = 0
    return dict_best_art


def get_user_articles(user_id, articles_df):
    user_index = articles_df.query(f'user_id == {user_id}').index
    user_articles = articles_df['article_id'].loc[user_index]
    user_articles = list(user_articles)
    return user_articles


def get_mean_vec_user(user_id, articles_df, model):
    user_index = articles_df.query(f'user_id == {user_id}').index
    articles = articles_df[['article_id', 'rating']].loc[user_index]
    articles = articles.to_records(index=False)
    articles = list(articles)
    articles.sort(key=lambda a: a[1], reverse=True)
    articles = [x[0] for x in articles[:5]]
    vectors = {}
    for article in articles:
        vectors[article] = model[article]
    vectors_df = pd.DataFrame(vectors)
    mean_vectors_df = vectors_df.mean(axis=1)
    return mean_vectors_df


def distance_nearest(user_id, articles_df, model):
    mean_articles_vec = get_mean_vec_user(user_id, articles_df, model)
    removed_articles = get_user_articles(user_id, articles_df)
    dist_dict = {}
    method = spatial.distance.cosine
    for article in range(len(model)):
        if article not in removed_articles:
            dist_dict[article] = method(mean_articles_vec, model[article])
        if len(dist_dict) > 5:
            lowest = [k for k, i in dist_dict.items() if i == max(dist_dict.values())]
            del dist_dict[lowest[0]]
    return dist_dict