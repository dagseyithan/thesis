from keras.layers import Conv1D
from keras.models import Sequential
import numpy as np
import pprint as pp
import text_utilities as tu
from data.datareader import readdata
from keras import backend as K
from elmo import __get_elmo_sentence_embedding, __get_elmo_word_embedding
from config.configurations import MAX_TEXT_WORD_LENGTH, ELMO_VECTOR_LENGTH, FASTTEXT_VECTOR_LENGTH


def get_combinations(vec_A, vec_B, max_text_length, window_size = 3):
    combined = []
    i, j = 0, 0
    while i+window_size <= max_text_length:
        while j+window_size <= max_text_length:
            stacked = np.vstack((vec_A[i:i+window_size], vec_B[j:j+window_size]))
            combined.append(list(stacked))
            j += 1
        j = 0
        i += 1
    return np.array(combined)


EMBEDDING_LENGTH = ELMO_VECTOR_LENGTH


input1 = np.zeros((MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH), dtype=float)
input2 = np.zeros((MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH), dtype=float)



text1 = "zwei wörter"
text2 = "und drei wörter"

text1_embedding = __get_elmo_sentence_embedding(text1)
#text1_embedding = [[1., 1., 1.], [12., 12., 12.], [13., 13., 13.],[14., 14., 14.], [15., 15., 15.], [16., 16., 16.]]
text1_word_length = len(text1.split())

text2_embedding = __get_elmo_sentence_embedding(text2)
#text2_embedding = [[2., 2., 2.], [22., 22., 22.], [23., 23., 23.], [24., 24., 24.], [25., 25., 25.], [26., 26., 26.]]
text2_word_length = len(text2.split())

input1[:text1_word_length] = text1_embedding[:MAX_TEXT_WORD_LENGTH]
input2[:text2_word_length] = text2_embedding[:MAX_TEXT_WORD_LENGTH]

#print(input1.size)
#combined = get_combinations(text1_embedding, text2_embedding, max_text_length = 6, embed_length = EMBEDDING_LENGTH, window_size = 3)
combined = get_combinations(input1, input2, max_text_length = MAX_TEXT_WORD_LENGTH, window_size = 3)

pp.pprint(combined)




'''

Data = readdata()
ProductX, ProductY = Data[Data.columns[0]], Data[Data.columns[4]]

print(ProductX[0])
text, extracted, numerals = tu.pre_process(ProductX[0])
print(text)
embeddingsent = __get_elmo_sentence_embedding("zwei wörter")

sent1 = embeddingsent
sent1 = np.expand_dims(sent1, axis=0)
print(sent1.shape)

model = Sequential()

conv = Conv1D(filters=1, kernel_size=2, kernel_initializer='truncated_normal', input_shape=(None, 1024), use_bias=True, activation=None, padding='same')

model.add(conv)

model.compile(optimizer='rmsprop', loss='mean_squared_error')
model.summary()

get_layer_output = K.function([model.layers[0].input], [model.layers[0].output])


layer_output = get_layer_output([sent1])[0]

print(layer_output)
print(layer_output.shape)

'''