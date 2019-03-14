from keras.layers import Dense, Conv3D
from keras.models import Sequential
from text_utilities import get_ngrams, get_fasttext_similarity, get_elmo_similarity


model = Sequential()

print('fastText:')
print(get_fasttext_similarity( '2014', 'year'))
print(get_fasttext_similarity( 'MVP', 'year'))
print(get_fasttext_similarity( 'win', 'won'))
print('elmo:')
print(get_elmo_similarity( '2014', 'year'))
print(get_elmo_similarity( 'MVP', 'year'))
print(get_elmo_similarity( 'win', 'won'))

