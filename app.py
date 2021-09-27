from flask import Flask, jsonify, request
import gensim
from sklearn.decomposition import PCA


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

model = gensim.models.Word2Vec.load("./data/ja.bin")


@app.route("/word2vec/words", methods=["GET"])
def words():
    data = {"words": model.wv.index2word}
    return jsonify(data)


@app.route("/word2vec/vec", methods=["GET"])
def word2vec():
    if request.args.get("words") is None:
        return jsonify({"message": "words is not specified"}), 400
    
    words = request.args.get("words").split(",")
    vec_list = []
    for word in words:
        if word not in model.wv:
            return jsonify({"message": "{} is not defined".format(word)}), 400
        vec_list.append(model.wv[word])

    pca = PCA(n_components=2)
    pca.fit(vec_list)
    vec_list_pca = pca.transform(vec_list)
    
    result = []
    for i in range(len(words)):
        word = words[i]
        vec = vec_list[i]
        vec_pca = vec_list_pca[i]
        result.append({
            "word": word,
            # "vec": vec.tolist(),
            "pca": vec_pca.tolist()
        })
    
    return jsonify({"code": 0, "result": result})


@app.route("/word2vec/similar", methods=["GET"])
def similar():
    # pos, neg
    if request.args.get("pos") is None and request.args.get("neg") is None:
        return jsonify({"message": "pos/neg is not specified"}), 400
    
    vec_list = []
    
    if request.args.get("pos") is None:
        pos = []
    else:
        pos = request.args.get("pos").split(",")
    
    if request.args.get("neg") is None:
        neg = []
    else:
        neg = request.args.get("neg").split(",")

    for word in pos:
        if word not in model.wv:
            return jsonify({"message": "{} is not defined".format(word)}), 400
        else:
            vec_list.append(model.wv[word])

    for word in neg:
        if word not in model.wv:
            return jsonify({"message": "{} is not defined".format(word)}), 400
        else:
            vec_list.append(model.wv[word])
    
    sims = model.wv.most_similar(positive=pos, negative=neg)
    for sim in sims:
        vec_list.append(model.wv[sim[0]])

    pca = PCA(n_components=2)
    pca.fit(vec_list)
    vec_list_pca = pca.transform(vec_list)

    input_pos = []
    input_neg = []
    result = []
    index = 0
    for word in pos:
        input_pos.append({
            "word": word,
            # "vec": model.wv[word].tolist(),
            "pca": vec_list_pca[index].tolist()
        })
        index += 1

    for word in neg:
        input_neg.append({
            "word": word,
            # "vec": model.wv[word].tolist(),
            "pca": vec_list_pca[index].tolist()
        })
        index += 1
    
    for sim in sims:
        result.append({
            "word": sim[0],
            "score": sim[1],
            # "vec": model.wv[sim[0]].tolist(),
            "pca": vec_list_pca[index].tolist()
        })
        index += 1
    
    return jsonify({"code": 1, "pos": input_pos, "neg": input_neg, "result": result})


if __name__ == "__main__":
    app.run()
