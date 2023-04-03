from surprise import dump
import pandas as pd
import os

model_filename = "./api/model/model.pickle"

# API version
__version__ = "0.1.0"

articles_df = pd.read_csv('./api/model/article.csv')

file_name = os.path.expanduser(model_filename)
_, model = dump.load(file_name)


def predict_articles(model, articles_df, user_id):
    predictions = {}
    articles = articles_df['article_id'].unique()
    user_article = list(articles_df['article_id'][articles_df['user_id'] == user_id])
    for u_article in user_article:
        articles.remove(u_article)
    for article in articles:
        predictions[model.predict(user_id, int(article)).iid] = model.predict(user_id, int(article)).est
    return predictions
