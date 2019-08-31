from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input, concatenate, Bidirectional, LSTM, Reshape, Layer, \
    ReLU, Lambda, Conv1D, add, AveragePooling2D, BatchNormalization
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, LambdaCallback, EarlyStopping, ReduceLROnPlateau
import keras.backend as K
from config import configurations
from data_utilities.generator import Native_DataGenerator_for_SemanticSimilarityNetwork_STS, \
    Native_DataGenerator_for_SemanticSimilarityNetwork_SICK
import numpy as np
from texttovector import get_ready_vector, get_ready_vector_on_batch
import time
from data_utilities.datareader import read_sts_data, read_sick_data
from sklearn.preprocessing import minmax_scale
import tensorflow as tf
import os

DIM = 9
MAX_TEXT_WORD_LENGTH = configurations.MAX_TEXT_WORD_LENGTH #10
MAX_WORD_CHARACTER_LENGTH = configurations.MAX_WORD_CHARACTER_LENGTH #60
EMBEDDING_LENGTH = configurations.EMBEDDING_LENGTH #300
ALPHABET_LENGTH = configurations.ALPHABET_LENGTH #54
WORD_TO_WORD_COMBINATIONS = int(MAX_TEXT_WORD_LENGTH * MAX_TEXT_WORD_LENGTH) #10*10
WORD_TENSOR_DEPTH = int((ALPHABET_LENGTH * MAX_WORD_CHARACTER_LENGTH) / DIM) #(54*60/9 = 360)
BATCH_SIZE = configurations.BATCH_SIZE

selected_activation='relu'

def tf_pearson(y_true, y_pred):
    return tf.contrib.metrics.streaming_pearson_correlation(y_pred, y_true)[1]

TRAIN = True




def SemanticSimilarityNetwork_Uni():

    embedded_sentence_A = Input(shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH))
    embedded_sentence_B = Input(shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH))

    conv1D_2n = Conv1D(filters=40, kernel_size=2, kernel_initializer='glorot_uniform',
                       bias_initializer='glorot_uniform',input_shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH),
                       use_bias=True, activation=selected_activation, padding='same', name='conv1D_2n')
    conv1D_3n = Conv1D(filters=40, kernel_size=3, kernel_initializer='glorot_uniform',
                       bias_initializer='glorot_uniform',input_shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH),
                       use_bias=True, activation=selected_activation,padding='same', name='conv1D_3n')
    conv2D_2x2 = Conv2D(filters=40, kernel_size=(2, 2), kernel_initializer='glorot_uniform',
                        bias_initializer='glorot_uniform', input_shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH),
                        use_bias=True, activation=selected_activation,
                        padding='same', name='conv2D_2x2')
    conv2D_3x3 = Conv2D(filters=40, kernel_size=(3,3), kernel_initializer='glorot_uniform',
                        bias_initializer='glorot_uniform', input_shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH),
                        use_bias=True, activation=selected_activation, padding='same', name='conv2D_3x3')
    maxPool2D_2x2 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_first',
                                 name='maxPool2D_2x2')
    averagePool2D_2x2 = AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_first',
                                         name='averagePool2D_2x2')
    expand_dims = Lambda(lambda x: K.expand_dims(x, axis=1))


    def bi_lstms_network():
        layers = [Bidirectional(LSTM(units=50, activation='tanh', return_sequences=True,
                                     kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform',
                                     input_shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH)), merge_mode='concat'),
                  Bidirectional(LSTM(units=100, activation='tanh', return_sequences=True,
                                     kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform',
                                     input_shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH)), merge_mode='concat')
                 ]

        def shared_layers(x):
            for layer in layers:
                x = layer(x)
            return x
        return shared_layers

    bi_lstms = bi_lstms_network()

    conv1D_2n_out_A = conv1D_2n(embedded_sentence_A)
    conv1D_3n_out_A = conv1D_3n(embedded_sentence_A)
    conv2D_2x2_out_A = conv2D_2x2(expand_dims(embedded_sentence_A))
    conv2D_3x3_out_A = conv2D_3x3(expand_dims(embedded_sentence_A))
    bi_lstms_out_A = bi_lstms(embedded_sentence_A)

    conv1D_2n_out_B = conv1D_2n(embedded_sentence_B)
    conv1D_3n_out_B = conv1D_3n(embedded_sentence_B)
    conv2D_2x2_out_B = conv2D_2x2(expand_dims(embedded_sentence_B))
    conv2D_3x3_out_B = conv2D_3x3(expand_dims(embedded_sentence_B))
    bi_lstms_out_B = bi_lstms(embedded_sentence_B)

    concat_conv1D_2n = concatenate([conv1D_2n_out_A, conv1D_2n_out_B], axis=-1)
    concat_conv1D_3n = concatenate([conv1D_3n_out_A, conv1D_3n_out_B], axis=-1)
    concat_2D_2x2 = concatenate([conv2D_2x2_out_A,  conv2D_2x2_out_B], axis=-1)
    concat_2D_3x3 = concatenate([conv2D_3x3_out_A, conv2D_3x3_out_B], axis=-1)
    concat_bilstms = concatenate([bi_lstms_out_A, bi_lstms_out_B], axis=-1)

    pooled_1D_2n = Flatten()(maxPool2D_2x2(expand_dims(concat_conv1D_2n)))
    pooled_1D_3n = Flatten()(maxPool2D_2x2(expand_dims(concat_conv1D_3n)))
    pooled_2D_2x2 = Flatten()(maxPool2D_2x2(concat_2D_2x2))
    pooled_2D_3x3 = Flatten()(maxPool2D_2x2(concat_2D_3x3))
    pooled_lstms = Flatten()(averagePool2D_2x2(expand_dims(concat_bilstms)))

    out =  Lambda(lambda x: K.concatenate(x, axis=-1))([pooled_1D_2n, pooled_1D_3n, pooled_2D_2x2, pooled_2D_3x3,
                                                   pooled_lstms])

    #out = MLP_score(out)

    model = Model(inputs=[embedded_sentence_A, embedded_sentence_B], outputs=out)

    return model

