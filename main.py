from flask import Flask, jsonify
from model import predict_articles, articles_df, model
import os

app = Flask(__name__)

@app.route('/')
def home():
    return '<p>Recommendation model</p>'


@app.route('/api/<user_id>')
def prediction(user_id):
    # return prediction in json
    return jsonify(user=user_id, pred=predict_articles(model, articles_df, user_id))


# to run the app in a docker and access to it
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
