import text_utilities as tu
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
import nltk
import datetime
from config import configurations
import numpy as np


stemmer = SnowballStemmer("german")

nltk.download('udhr')
nltk.download('udhr2')
nltk.download('punkt')
nltk.download('stopwords')

stop_words = nltk.corpus.stopwords.words('german') + list(punctuation)

corp = nltk.corpus.ConllCorpusReader('data/', configurations.TIGER_CORPUS_FILE, ['ignore', 'words', 'ignore', 'ignore', 'pos'], encoding='utf-8')
words = list(corp.words())
sents = list(corp.sents())

vocabulary = set()
words = [w.lower() for w in words]
vocabulary.update([w for w in words if w not in stop_words and not w.isdigit()])

vocabulary = list(vocabulary)
word_index = {w: idx for idx, w in enumerate(vocabulary)}

VOCABULARY_SIZE = len(vocabulary)
DOCUMENTS_COUNT = len(corp.sents())

print(VOCABULARY_SIZE, DOCUMENTS_COUNT)

word_idf = np.zeros(VOCABULARY_SIZE)
words = vocabulary
indexes = [word_index[word] for word in words]
word_idf[indexes] += 1.0
word_idf = np.log(DOCUMENTS_COUNT / (1 + word_idf).astype(float))
print(word_idf[word_index['deutlich']])
print(word_idf[word_index['tag']])

tfidf = TfidfVectorizer(tokenizer=tu.word_tokenize, stop_words=stop_words, decode_error='ignore')
print('building term-document matrix... [process started: ' + str(datetime.datetime.now()) + ']')
tdm = tfidf.fit_transform([corp.raw()])
print('done! [process finished: ' + str(datetime.datetime.now()) + ']')
print(tdm.shape)

feature_names = tfidf.get_feature_names()
print('TDM contains ' + str(len(feature_names)) + ' terms and ' + str(tdm.shape[0]) + ' documents')

print(tdm[0, tfidf.vocabulary_['angela']]*1000)
print(tdm[0, tfidf.vocabulary_['merkel']]*1000)
print(tdm[0, tfidf.vocabulary_['türkei']]*1000)
print(tdm[0, tfidf.vocabulary_['deutschland']]*1000)


'''

def tokenize_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


token_dict = {}
print(udhr.fileids())
for article in udhr.fileids():
    if article == 'German_Deutsch-Latin1':
        token_dict[article] = udhr.raw(article)

tfidf = TfidfVectorizer(tokenizer=tokenize_and_stem, stop_words=None, decode_error='ignore')
print('building term-document matrix... [process started: ' + str(datetime.datetime.now()) + ']')
sys.stdout.flush()

tdm = tfidf.fit_transform(token_dict.values())
print('done! [process finished: ' + str(datetime.datetime.now()) + ']')

print(tdm.shape)
feature_names = tfidf.get_feature_names()
print('TDM contains ' + str(len(feature_names)) + ' terms and ' + str(tdm.shape[0]) + ' documents')

print('first term: ' + feature_names[0])
print('last term: ' + feature_names[len(feature_names) - 1])

for i in range(0, 4):
    print('random term: ' + feature_names[randint(1,len(feature_names) - 2)])

print('fastText:')
print(tu.get_fasttext_word_similarity( '2014', 'year'))
print(tu.get_fasttext_word_similarity( 'MVP', 'year'))
print(tu.get_fasttext_word_similarity( 'win', 'won'))
print('elmo:')
print(tu.get_elmo_word_similarity( '2014', 'year'))
print(tu.get_elmo_word_similarity( 'MVP', 'year'))
print(tu.get_elmo_word_similarity( 'win', 'won'))

print(tu.get_ngrams('doch das wäre echt gut', n=2))

print()

'''