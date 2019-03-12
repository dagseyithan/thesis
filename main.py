from keras.layers import Dense, Conv3D
from keras.models import Sequential
from fasttext import get_fasttext_embedding

model = Sequential()


print(get_fasttext_embedding('Ich'))
