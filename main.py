from keras.layers import Dense, Conv3D
from keras.models import Sequential
from fasttext import get_fasttext_embedding
from elmo import get_elmo_embedding
from text_utilities import get_ngrams
from scipy.spatial import distance

model = Sequential()


print(1.0 - distance.cosine(get_fasttext_embedding('gehen'), get_fasttext_embedding('gehen')))
print(1.0 - distance.cosine(get_fasttext_embedding('gehen'), get_fasttext_embedding('ging')))
print(1.0 - distance.cosine(get_elmo_embedding('gehen'), get_elmo_embedding('gehen')))
print(1.0 - distance.cosine(get_elmo_embedding('gehen'), get_elmo_embedding('ging')))
