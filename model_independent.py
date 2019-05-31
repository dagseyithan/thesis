from keras.layers import Conv1D, Conv2D, BatchNormalization, MaxPooling2D, Dense, Reshape, Flatten, Input, concatenate, \
    Lambda, CuDNNGRU, AveragePooling3D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
from config.configurations import MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH, BATCH_SIZE
from data_utilities.generator import Native_DataGenerator_for_IndependentModel
from keras.utils import plot_model
from texttovector import get_ready_vector
import numpy as np

TRAIN = True


def hinge_loss(y_true, y_pred, alpha=1.0):
    slice_pos = lambda x: x[0:BATCH_SIZE, :]
    slice_neg = lambda x: x[BATCH_SIZE:BATCH_SIZE * 2, :]

    positive = Lambda(slice_pos, output_shape=(BATCH_SIZE, 1))(y_pred)
    negative = Lambda(slice_neg, output_shape=(BATCH_SIZE, 1))(y_pred)

    basic_loss = alpha + negative - positive

    loss = K.mean(K.maximum(basic_loss, 0.0), axis=-1)

    return loss



input_a = Input(shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH))
input_b = Input(shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH))
conv_out = Conv1D(filters=400, kernel_size=3, kernel_initializer='glorot_uniform',
                  input_shape=(None, EMBEDDING_LENGTH), use_bias=True, activation='relu', padding='same')
conv_out_a = conv_out(input_a)
conv_out_b = conv_out(input_b)
gru_a = CuDNNGRU(units=100, return_sequences=True, input_shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH))
gru_b = CuDNNGRU(units=100, return_sequences=True, input_shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH))
gru_a_out_a = gru_a(conv_out_a)
gru_a_out_b = gru_a(conv_out_b)
gru_b_out_a = gru_b(conv_out_a)
gru_b_out_b = gru_b(conv_out_b)
concat_a = concatenate([gru_a_out_a, gru_b_out_a], axis=1)
concat_b = concatenate([gru_a_out_b, gru_b_out_b], axis=1)
concat_a = Reshape((1, 2, MAX_TEXT_WORD_LENGTH, 100))(concat_a)
concat_b = Reshape((1, 2, MAX_TEXT_WORD_LENGTH, 100))(concat_b)
avgpool3ded_a = AveragePooling3D(pool_size=(2, 1, 1), strides=None, padding='valid',
                                 input_shape=(1, 2, MAX_TEXT_WORD_LENGTH, 100), data_format='channels_first')(concat_a)
avgpool3ded_b = AveragePooling3D(pool_size=(2, 1, 1), strides=None, padding='valid',
                                 input_shape=(1, 2, MAX_TEXT_WORD_LENGTH, 100), data_format='channels_first')(concat_b)

avgpool3ded_a = Reshape((1, MAX_TEXT_WORD_LENGTH, 100))(avgpool3ded_a)
avgpool3ded_b = Reshape((1, MAX_TEXT_WORD_LENGTH, 100))(avgpool3ded_b)


def common_network():
    layers = [Conv2D(filters=100, kernel_size=(3, 3), kernel_initializer='glorot_uniform',
                     input_shape=(1, MAX_TEXT_WORD_LENGTH, 100), data_format='channels_first',
                     use_bias=True,activation='relu', padding='same'),
              MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid',data_format='channels_first'),
              Conv2D(filters=100, kernel_size=(3, 3), kernel_initializer='glorot_uniform',
                     data_format='channels_first', use_bias=True,
                     activation='relu', padding='same'),
              MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_first')
    ]

    def shared_layers(x):
        for layer in layers:
            x = layer(x)
        return x

    return shared_layers

common_net = common_network()
a = common_net(avgpool3ded_a)
b = common_net(avgpool3ded_b)

concat_ab = concatenate([a, b], axis=-1)

x = Flatten()(concat_ab)
x = Dense(activation='relu', units=128, use_bias=True)(x)
x = Dense(activation='relu', units=64, use_bias=True)(x)
x = Dense(activation='relu', units=32, use_bias=True)(x)
x = Dense(activation='relu', units=16, use_bias=True)(x)
out = Dense(activation='softplus', units=1, use_bias=True)(x)

net = Model([input_a, input_b], out)
net.summary()
plot_model(net, to_file='subnet.png', show_layer_names=True)

anchor_in = Input(shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH))
pos_in = Input(shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH))
neg_in = Input(shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH))

anchor_pos_out = net([anchor_in, pos_in])
anchor_neg_out = net([anchor_in, neg_in])

net_out = concatenate([anchor_pos_out, anchor_neg_out], axis=0)

model = Model(inputs=[anchor_in, pos_in, neg_in], outputs=net_out)
model.compile(optimizer=Adam(lr=0.0001), loss=hinge_loss)
model.summary()
plot_model(model, to_file='model.png', show_layer_names=True)


if TRAIN:
    # model = load_model('trained_models/model_arc2_02_concat.h5', custom_objects={'hinge_loss': hinge_loss})
    data_generator = Native_DataGenerator_for_IndependentModel(batch_size=BATCH_SIZE)
    model.fit_generator(generator=data_generator, shuffle=True, epochs=25, workers=1, use_multiprocessing=False)
    model.save('trained_models/model_independent_00_FastText.h5')
else:
    model = load_model('trained_models/model_independent_00_FastText.h5', custom_objects={'hinge_loss': hinge_loss})
    model.summary()


def get_similarity_arc2(textA, textB):
    return 0