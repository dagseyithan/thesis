from keras.layers import Conv1D
from keras.models import Sequential
from keras import backend as K
import numpy as np

#sent1: "I go there", assumption: each word has (1, 4) dimension embeddings
#sent1 = [[e0_w0, e1_w0, e2_w0], ...]

sent1 = np.array([[0., 0., 0., 1], [1., 1., 1., 2]])
sent1 = np.expand_dims(sent1, axis=0)
print(sent1.shape)

sent2 = np.array([0., 0., 0., 1., 1., 1., 2., 2., 2.])
sent2 = np.expand_dims(sent2, axis=0)

print(sent2.shape)

model = Sequential()


conv = Conv1D(filters=1, kernel_size=2, kernel_initializer='ones', input_shape=(2, 4), use_bias=True, activation=None)

model.add(conv)

model.compile(optimizer='rmsprop', loss='mean_squared_error')
model.summary()

get_layer_output = K.function([model.layers[0].input], [model.layers[0].output])


layer_output = get_layer_output([sent1])[0]

print(layer_output)
