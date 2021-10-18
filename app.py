from flask import Flask
from flask_cors import CORS
from blueprints.word2vec import word2vec_bp


app = Flask(__name__)
CORS(app)
app.config['JSON_AS_ASCII'] = False

app.register_blueprint(word2vec_bp)


if __name__ == "__main__":
    app.run()
