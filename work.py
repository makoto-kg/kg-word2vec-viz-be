import gensim

# model = gensim.models.KeyedVectors.load_word2vec_format('./data/ja.bin', binary=True)
model = gensim.models.Word2Vec.load("./data/ja.bin")

# print(model.wv["日本"])
# print(model.wv.most_similar("日本"))
# print(model.wv.most_similar(positive=['姪', '男性'], negative=['女性']))
# print(model.wv.index2word)