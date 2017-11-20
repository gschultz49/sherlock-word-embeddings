import sys
from nltk import word_tokenize
from gensim.models import Word2Vec

# Run this file to generate a word vectors file of the selected type
class Embedding_Maker(object):
    
    def __init__(self, source_file_path, vector_name, embedding_type):
        assert (embedding_type == "skip" or embedding_type == "cbow"), "Invalid embedding type, please use 'cbow' (Continuous bag of Words) or 'skip' (Skipgram)"
        assert (vector_name[-4:] == ".bin"), "Invalid vector output format '{}', please use '.bin' instead".format(vector_name)

        print ("Generating '{}' type word embeddings from '{}' input file...".format(embedding_type, source_file_path))

        self.source_name = str(source_file_path)
        self.vector_name = str(vector_name)
        self.embedding_type = str(embedding_type).lower()
        
        self.sentences = self.read_and_convert_to_sentences(self.source_name)
        self.create_model(self.sentences, self.vector_name, self.embedding_type)

    def read_and_convert_to_sentences(self, source_name):
        sentences = []
        # source file name defined here
        reader = open(source_name)
        # read the rest of the lines of the file
        for line in reader:
            # get the next line of the metadata file, and split it into columns
            line = line.rstrip()
            tokens = [x.lower() for x in word_tokenize(line)]
            # skips over empty lines
            if len(tokens) > 1 :
                sentences.append(tokens)
        return sentences

    # uses gensim/google Word2Vec with CBOW or Skipgram method to create the word embeddings
    def create_model(self, sentences, vector_name, embedding_type):
        # use the skipgram type
        if embedding_type == "skip":
            model = Word2Vec(sentences, min_count = 1, sg=1)
        # or a cbow type
        else: 
            model = Word2Vec(sentences, min_count = 1, sg=0)
        model.save(vector_name)
        print ("Word embeddings saved to '{}'".format(vector_name))


if __name__ == '__main__':
    assert (len(sys.argv) > 3), "Please enter arguements for the <source_file>, <vector_name> and <embedding_type>!"
    # python embeddings.py <source_file> <vector_name.bin> <'cbow' or 'skip'>
    Embedding_Maker(sys.argv[1], sys.argv[2], sys.argv[3])

