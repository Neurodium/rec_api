from flask import Flask, jsonify
from model import distance_nearest, articles_df, model, get_best_articles
import os

app = Flask(__name__)

@app.route('/')
def home():
    return '<p>Recommendation model</p>'


@app.route('/api/<user_id>')
def prediction(user_id):
    # return prediction in json
    try:
        pred = distance_nearest(user_id, articles_df, model)
        return jsonify(user=user_id, pred=pred)
    except:
        return jsonify(user=user_id, pred=get_best_articles())


# to run the app in a docker and access to it
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
