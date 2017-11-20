from gensim.models import Word2Vec

def load_vector_file(filename):
    reader = Word2Vec.load(filename)
    return reader

def most_similar_word(word):
    return model.wv.most_similar(positive =[word.lower().rstrip()])

if __name__ == "__main__":
    # model = load_vector_file("vectors_lowercase_cbow.bin")
    model = load_vector_file("vectors_lowercase_skipgram.bin")
