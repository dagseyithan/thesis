from nltk import ngrams



def get_ngrams(text, n=1):
    '''
    function to get ngrams of a given text.
    :param text: text to get the ngram of.
    :param n: gram count. default = 1.
    :return: returns the list of the lists of n-grams of the given text.
    '''

    return list(list(gram) for gram in ngrams(text.split(), n))