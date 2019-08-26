from nltk import ngrams
from nltk import word_tokenize
from string import punctuation
from nltk.corpus import stopwords
from config import configurations
from fasttext import __get_fasttext_word_embedding, __get_fasttext_sentence_embedding
#from elmo import __get_elmo_word_embedding
from quantulum import load as l
from scipy.spatial import distance
from secos.decompound import split_compounds
from word2number import w2n
import nltk
import numpy as np

nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

if configurations.LANGUAGE == 'GERMAN':
    stop_words = stopwords.words('german') + list(punctuation)
elif configurations.LANGUAGE == 'ENGLISH':
    stop_words = stopwords.words('english') + list(punctuation)

TOTAL_VEC_LENGTH = 360
DIC_VEC_LENGTH = 50
NUM_VEC_LENGTH = 10

dict_path = '/home/sdag/Schreibtisch/data/german_words_dictionary/german.dic'


LOWERCASE_UNITS = [w.lower() for w in l.UNITS.keys()]
GERMAN_PREPOSITIONS = ['über', 'oben', 'nach', 'gegen', 'unter', 'um', 'wie', 'bei', 'vor', 'hinter', 'unten', 'unter',
                       'neben', 'zwischen', 'darüber hinaus', 'aber', 'durch', 'trotz', 'nach unten', 'während',
                       'ausgenommen', 'für', 'von', 'in', 'innen', 'in der Nähe von', 'nächste', 'von', 'auf',
                       'gegenüber', 'heraus', 'außerhalb', 'über', 'pro', 'plus', 'seit', 'als', 'bis', 'hinauf',
                       'mit', 'innerhalb', 'ohne', 'zu', 'zum', 'zur']


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

'''
def get_elmo_word_similarity(worda = None, wordb = None):
    if worda == None or wordb == None:
        print('comparison with null value(s)!')
        return None
    else:
        return 1.0 - distance.cosine(__get_elmo_word_embedding(worda), __get_elmo_word_embedding(wordb))
'''

def tokenize(text):
    words = word_tokenize(text)
    words = [w.lower() for w in words]
    return [w for w in words if w not in stop_words and not w.isdigit()]


def save_tfidf_scores(vectorizer, matrix):
    scores_dict = {}
    scores = zip(vectorizer.get_feature_names(),
                 np.asarray(matrix.sum(axis=0)).ravel())
    #sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    for item in scores:
        scores_dict[item[0]] = item[1]
    return scores_dict


def get_ngrams(text, n=1):
    '''
    function to get ngrams of a given text.
    :param text: text to get the ngram of.
    :param n: gram count. default = 1.
    :return: returns the list of the lists of n-grams of the given text.
    '''

    return list(list(gram) for gram in ngrams(text.split(), n))

def get_dictionary_index(word):
    with open(dict_path, 'r', encoding = 'ISO-8859-1') as f:
        words_dictionary = f.read().splitlines()
    word = word.lower()
    try:
        return words_dictionary.index(word)
    except ValueError:
        try:
            return words_dictionary.index(word.title())
        except ValueError:
            try:
                return words_dictionary.index(word.upper())
            except ValueError:
                return 0


def remove_single_characters(text):
    clean = []
    for w in text.split():
        if len(w) > 1:
            clean.append(w)
        else:
            if w.isdigit():
                clean.append(w)
    return ' '.join([w for w in clean])


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def get_number_of_elements(text):
    return len(text.split())


def has_german_prepositions(text):
    return True if len([w for w in text.split() if w in GERMAN_PREPOSITIONS]) > 0 else False


def has_units(text):
    return True if len([w for w in text.split() if w in LOWERCASE_UNITS]) > 0 else False


def remove_german_prepositions(text):
    return ' '.join([w for w in text.split() if w not in GERMAN_PREPOSITIONS])


def remove_units(text):
    return ' '.join([w for w in text.split() if w not in LOWERCASE_UNITS])


def remove_minthree(text):
    clean = []
    for w in text.split():
        if len(w) >= 3:
            clean.append(w)
        else:
            if is_number(w):
                clean.append(w)
    return ' '.join([w for w in clean])


def get_number_of_common_words(texta, textb):
    return len([w for w in texta.split() if w in textb.split()]) if len([w for w in texta.split() if w in textb.split()]) > 1 else 1


def align_texts(texta, textb):
    '''
    function to align textb to texta.
    :param texta:
    :param textb:
    :return: returns aligned textb along with texta again, for ease of use.
    '''

    textbe = ''

    for word in texta.split():
        if word in textb.split():
            textbe += word + ' '
            textb = textb.replace(word + ' ', '')
    textb = textb.lstrip()
    textbe = textbe.lstrip()
    textbe += textb

    return texta, textbe


def pre_process(text):
    '''
    function for preprocessing given string.
    :param text: strıng to be preprocessed
    :return: preprocessed string
    '''
    text = text.lower()
    text = remove_punctuation(text)
    text = separate_numerals(text)
    print(text)
    #text = remove_single_characters(text)
    #if has_units(text):
        #text = remove_units(text)
    text = split_compounds(text)
    simply_processed_text = text
    text, numerals = extract_numerals(text)
    extracted = text
    if text == '' or text == ' ': #extremely rare but sometimes nltk.postagger missclassifies POS tags, leading to erroneous extraction.
        return ''
    for numeral in numerals:
        text = text + ' ' + str(numeral)

    return simply_processed_text, extracted, numerals

def pre_process_single_return(text):
    '''
    function for preprocessing given string.
    :param text: strıng to be preprocessed
    :return: preprocessed string
    '''
    text = text.lower()
    text = remove_punctuation(text)
    text = separate_numerals(text)
    #text = remove_single_characters(text)
    #if has_units(text):
        #text = remove_units(text)
    text = split_compounds(text)
    simply_processed_text = text
    text, numerals = extract_numerals(text)
    extracted = text
    if text == '' or text == ' ': #extremely rare but sometimes nltk.postagger missclassifies POS tags, leading to erroneous extraction.
        return ''
    for numeral in numerals:
        text = text + ' ' + str(numeral)

    return simply_processed_text


def get_single_average_sentence_vector(text):
    return np.array(__get_fasttext_sentence_embedding(text))


def get_single_average_sentence_vector_without_numerals(text):
    Extracted = extract_numerals(text)
    return np.array(__get_fasttext_sentence_embedding(' '.join([w for w in Extracted[0]])))


def get_combined_vector(texta, textb):
    Vec = np.array([])
    CombinedVec = np.array([])
    texta = remove_punctuation(texta)
    textb = remove_punctuation(textb)
    texta = separate_numerals(texta)
    textb = separate_numerals(textb)
    Vec = np.append(Vec, __get_fasttext_sentence_embedding(texta))
    Extracted = extract_numerals(texta)
    Vec = np.append(Vec, np.pad(create_dictionary_indexes_vector(np.array(Extracted[0])), (0, abs(DIC_VEC_LENGTH - len(Extracted[0]))), 'constant'))
    Vec = np.append(Vec, np.pad(np.array(Extracted[1], dtype=float), (0, abs(NUM_VEC_LENGTH - len(Extracted[1]))), 'constant')[0:10])
    Vec = np.reshape(Vec, (1, TOTAL_VEC_LENGTH))
    CombinedVec = np.append(CombinedVec, Vec)
    Vec = np.array([])
    Vec = np.append(Vec, __get_fasttext_sentence_embedding(textb))
    Extracted = extract_numerals(textb)
    Vec = np.append(Vec, np.pad(create_dictionary_indexes_vector(np.array(Extracted[0])), (0, abs(DIC_VEC_LENGTH - len(Extracted[0]))), 'constant'))
    Vec = np.append(Vec, np.pad(np.array(Extracted[1], dtype=float), (0, abs(NUM_VEC_LENGTH - len(Extracted[1]))), 'constant')[0:10])
    Vec = np.reshape(Vec, (1, TOTAL_VEC_LENGTH))
    CombinedVec = np.append(CombinedVec, Vec)
    return CombinedVec


def create_dictionary_indexes_vector(word_list):
    return [get_dictionary_index(w) for w in word_list]


def remove_punctuation(text):
    return ''.join([w if w not in punctuation else ' ' for w in text]).replace(u'\u00AE', '').replace(u'\u2122', '') #remove registered trademark (u00AE)/trademark (u2122) signs


def separate_numerals(text):
    '''
    function for separating numerals stuck to substrings in a given text.
    '''
    c = 0
    while c < len(text) - 1:
        if text[c].isdigit() and c <= len(text) - 1:
            if text[c + 1].isdigit() is False and text[c + 1] is not ' ':
                text = text[:c + 1] + ' ' + text[c + 1:]
        elif text[c].isdigit() is False and c <= len(text) - 1 and text[c] is not ' ':
            if text[c + 1].isdigit() is True:
                text = text[:c + 1] + ' ' + text[c + 1:]
        c = c + 1

    return text


def extract_numerals(text):
    '''
    function for extracting numerals in a given sentence.
    :return = str, []: text stripped of numerals, [numerals]
    '''
    extracted = [[], []]

    #text = separate_numerals(text)
    [extracted[0].append(w[0]) for w in nltk.pos_tag(text.split()) if w[1] != 'CD']
    [extracted[1].append(w[0]) for w in nltk.pos_tag(text.split()) if w[1] == 'CD']
    i = 0
    while i < len(extracted[1]):
        try:
            extracted[1][i] = w2n.word_to_num(extracted[1][i])
        except ValueError:
            extracted[1].remove(extracted[1][i])
        i = i + 1
    return ' '.join([w for w in extracted[0]]), extracted[1]







