from keras.layers import Conv1D, Conv2D, BatchNormalization, MaxPooling2D, Dense, Reshape, Flatten, Input
from keras.models import Sequential, Model
import numpy as np
from config.configurations import MAX_TEXT_WORD_LENGTH, ELMO_VECTOR_LENGTH, FASTTEXT_VECTOR_LENGTH
from data.generator import DataGenerator_for_Arc2


EMBEDDING_LENGTH = ELMO_VECTOR_LENGTH
COMBINATION_COUNT = 1944


def hinge_loss():
    return 0



def create_network(input_shape):

    model = Sequential()
    model.add(BatchNormalization(input_shape = input_shape))
    model.add(Conv1D(filters=100, kernel_size=3, kernel_initializer='truncated_normal', input_shape=(None, EMBEDDING_LENGTH), use_bias=True, activation='relu', padding='same'))
    model.add(Reshape((COMBINATION_COUNT, 10, 10)))
    model.add(Conv2D(filters=20, kernel_size=(3, 3), kernel_initializer='truncated_normal', input_shape=(None, EMBEDDING_LENGTH), data_format='channels_first', use_bias=True, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_first'))
    model.add(Conv2D(filters=100, kernel_size=(3, 3), kernel_initializer='truncated_normal', input_shape=(None, EMBEDDING_LENGTH), data_format='channels_first', use_bias=True, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=100, kernel_size=(3, 3), kernel_initializer='truncated_normal', input_shape=(None, EMBEDDING_LENGTH), data_format='channels_first', use_bias=True, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_first'))
    model.add(Flatten())
    model.add(Dense(activation='relu', units=64, use_bias=True))
    model.add(Dense(activation='relu', units=32, use_bias=True))
    model.add(Dense(activation='softplus', units=1, use_bias=True))

    return model


pos_in = Input(shape=(COMBINATION_COUNT, EMBEDDING_LENGTH))
neg_in = Input(shape=(COMBINATION_COUNT, EMBEDDING_LENGTH))

net = create_network(input_shape=(None, EMBEDDING_LENGTH))

pos_out = net(pos_in)
neg_out = net(neg_in)
net_out = [pos_out, neg_out]

model = Model(inputs=[pos_in, neg_in], outputs=net_out)
model.compile(optimizer='adam', loss='mse')

data_generator = DataGenerator_for_Arc2(batch_size=5)

model.fit_generator(generator=data_generator, shuffle=True, epochs=10)


