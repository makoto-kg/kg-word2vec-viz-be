import gensim
from sklearn.decomposition import PCA

# model = gensim.models.KeyedVectors.load_word2vec_format('./data/ja.bin', binary=True)
model = gensim.models.Word2Vec.load("./data/ja.bin")

# print(model.wv["日本"])
# print(model.wv.most_similar("日本"))
# print(model.wv.most_similar(positive=['姪', '男性'], negative=['女性']))
# print(model.wv.index2word)

sims = model.wv.most_similar(positive=['姪', '男性'], negative=['女性'])

for sim in sims:
    print(sim[0])

# data = []
# data.append(model.wv["日本"])
# data.append(model.wv["男性"])
# data.append(model.wv["女性"])
# data.append(model.wv["アメリカ"])

# pca = PCA(n_components=2)
# pca.fit(data)
# data_pca = pca.transform(data)

# print(data_pca)
