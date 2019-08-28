import numpy as np

def loadGloveModel():
    f = open(r'C:\Users\seyit\PycharmProjects\thesis\glove\glove.6B.50d.txt','r', encoding='utf-8')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print('glove50 has been loaded.')
    return model

model = loadGloveModel()

def __get_glove_word_embedding(text):
    try:
        return model[text]
    except KeyError:
        return np.zeros((50))


def __get_glove_average_sentence_embedding(sentence):
    return __get_glove_embeddings_average(__get_glove_sentence_embedding(sentence))


def __get_glove_sentence_embedding(sentence):
    return np.array([__get_glove_word_embedding(word) for word in sentence.split()])


def __get_glove_embeddings_average(sentence_vectors):
    num_words = np.float32(len(sentence_vectors))
    sentence_vectors = np.array(sentence_vectors)
    return np.divide(sentence_vectors.sum(axis=0), num_words)