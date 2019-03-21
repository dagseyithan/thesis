from nltk import ngrams
from nltk import word_tokenize
from string import punctuation
from nltk.corpus import stopwords
from config import configurations
from fasttext import __get_fasttext_word_embedding
from elmo import __get_elmo_word_embedding
from scipy.spatial import distance

if configurations.LANGUAGE == 'GERMAN':
    stop_words = stopwords.words('german') + list(punctuation)
elif configurations.LANGUAGE == 'ENGLISH':
    stop_words = stopwords.words('english') + list(punctuation)


def get_ngrams(text, n=1):
    '''
    function to get ngrams of a given text.
    :param text: text to get the ngram of.
    :param n: gram count. default = 1.
    :return: returns the list of the lists of n-grams of the given text.
    '''

    return list(list(gram) for gram in ngrams(text.split(), n))


def get_fasttext_word_similarity(worda = None, wordb = None):
    if worda == None or wordb == None:
        print('comparison with null value(s)!')
        return None
    else:
        return 1.0 - distance.cosine(__get_fasttext_word_embedding(worda), __get_fasttext_word_embedding(wordb))


def get_elmo_word_similarity(worda = None, wordb = None):
    if worda == None or wordb == None:
        print('comparison with null value(s)!')
        return None
    else:
        return 1.0 - distance.cosine(__get_elmo_word_embedding(worda), __get_elmo_word_embedding(wordb))


def tokenize(text):
    words = word_tokenize(text)
    words = [w.lower() for w in words]
    return [w for w in words if w not in stop_words and not w.isdigit()]




