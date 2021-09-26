from flask import Flask, jsonify
import gensim


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

model = gensim.models.Word2Vec.load("./data/ja.bin")


@app.route("/words", methods=["GET"])
def words():
    data = {"words": model.wv.index2word}
    return jsonify(data)


if __name__ == "__main__":
    app.run()
