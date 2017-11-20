# I think word embeddings are pretty cool

They have uses for downstream neural networks, or a deep reading of a particular corpus. While their results should always be taken with a grain of salt (https://mimno.infosci.cornell.edu/info3350/readings/antoniak.pdf), they can reveal noteworthy insights.
<br>
So, I decided to take my favorite collection of short stories and novels, the Sherlock Holmes collection, and generate word embeddings using the Google word2vec model with the help of gensim. 
<br>
There are two pre-trained word embeddings files in output/.bin which have vectors trained on this corpus using the Continuous Bag of Words (cbow) method, and the Skip-gram (skip) method. My personal insights from the corpus are within the output/.txt files. The source text for these vectors is from the USA version of https://sherlock-holm.es/ascii/. 

# Installation

As mentioned before, this generator uses `gensim` to create the word embeddings. It also uses `nltk` in order to parse the sentences into tokens.

`pip install gensim`
`pip install nltk`

# Usage

If you want to make your own vectors, you will need to pass in 3 arguments. These are:
<br>
1. A source file (i.e curated.txt)
2. An output name for your vector file (skipgram.bin). This needs to have a '.bin' file extension
3. An embedding type, either "cbow" or Continuous Bag of Words or "skip" for Skipgram.
<br>    
Then, run the command as follows:
`python embeddings.py <source_file> <output_name.bin> <'cbow' or 'skip'>`

For example:
`python embeddings.py "data/curated.txt" "skip_vectors.bin" "skip"`

# More food for thought

Word2Vec: https://deeplearning4j.org/word2vec.html
Skip gram methodology : http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
More on Gensim: https://radimrehurek.com/gensim/models/word2vec.html , https://machinelearningmastery.com/develop-word-embeddings-python-gensim/ 
Cosine Similarity: stackoverflow.com/questions/15173225/how-to-calculate-cosine-similarity-given-2-sentence-strings-python 


This project was made using Python 3.5.4