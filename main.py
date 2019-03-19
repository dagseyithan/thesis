from keras.layers import Dense, Conv3D
from keras.models import Sequential
import text_utilities as tu
from nltk.corpus import udhr
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
import nltk
import re
import datetime, sys
from random import randint


stemmer = SnowballStemmer("german")

nltk.download('udhr')
nltk.download('udhr2')
nltk.download('punkt')

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

print()

