from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input, concatenate, Bidirectional, LSTM, Reshape, Layer, \
    ReLU, Lambda, Conv1D, add, AveragePooling2D
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, LambdaCallback
import keras.backend as K
from config import configurations
import numpy as np
from encoder import encode_word, convert_to_tensor
from structural_similarity_network import StructuralSimilarityNetwork
import time

DIM = 9
MAX_TEXT_WORD_LENGTH = configurations.MAX_TEXT_WORD_LENGTH #10
MAX_WORD_CHARACTER_LENGTH = configurations.MAX_WORD_CHARACTER_LENGTH #60
EMBEDDING_LENGTH = configurations.FASTTEXT_VECTOR_LENGTH #300
ALPHABET_LENGTH = configurations.ALPHABET_LENGTH #54
WORD_TO_WORD_COMBINATIONS = int(MAX_TEXT_WORD_LENGTH * MAX_TEXT_WORD_LENGTH) #10*10
WORD_TENSOR_DEPTH = int((ALPHABET_LENGTH * MAX_WORD_CHARACTER_LENGTH) / DIM) #(54*60/9 = 360)
BATCH_SIZE = configurations.BATCH_SIZE


class MLPConv(Layer):
    def __init__(self, num_outputs):
        super(MLPConv, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        super(MLPConv, self).build(input_shape)
        self.dense_1 = Dense(50, activation='relu', kernel_initializer='glorot_uniform', use_bias=True)
        self.dense_2 = Dense(25, activation='relu', kernel_initializer='glorot_uniform', use_bias=True)
        self.dense_3 = Dense(5, activation='relu', kernel_initializer='glorot_uniform', use_bias=True)
        self.dense_4 = Dense(1, activation='relu', kernel_initializer='glorot_uniform', use_bias=True)

    def call(self, input):
        x = self.dense_1(input)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.dense_4(x)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)


class MLPLSTM(Layer):
    def __init__(self, num_outputs):
        super(MLPLSTM, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        super(MLPLSTM, self).build(input_shape)
        self.dense_1 = Dense(1000, activation='relu', kernel_initializer='glorot_uniform', use_bias=True)
        self.dense_2 = Dense(100, activation='relu', kernel_initializer='glorot_uniform', use_bias=True)
        self.dense_3 = Dense(10, activation='relu', kernel_initializer='glorot_uniform', use_bias=True)
        self.dense_4 = Dense(1, activation='relu', kernel_initializer='glorot_uniform', use_bias=True)

    def call(self, input):
        x = self.dense_1(input)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.dense_4(x)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)


def SemanticSimilarityNetwork():

    embedded_sentence_A = Input(shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH)) #(10, 300)
    embedded_sentence_B = Input(shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH)) #(10, 300)


    conv1D_2n = Conv1D(filters=10, kernel_size=2, kernel_initializer='glorot_uniform',
                       bias_initializer='glorot_uniform',input_shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH),
                       use_bias=True, activation='relu', padding='same', name='conv1D_2n')
    conv1D_3n = Conv1D(filters=10, kernel_size=3, kernel_initializer='glorot_uniform',
                       bias_initializer='glorot_uniform',input_shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH),
                       use_bias=True, activation='relu',padding='same', name='conv1D_3n')
    conv2D_2x2 = Conv2D(filters=10, kernel_size=(2, 2), kernel_initializer='glorot_uniform',
                        bias_initializer='glorot_uniform', input_shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH),
                        use_bias=True, activation='relu',
                        padding='same', name='conv2D_2x2')
    conv2D_3x3 = Conv2D(filters=10, kernel_size=(3,3), kernel_initializer='glorot_uniform',
                        bias_initializer='glorot_uniform', input_shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH),
                        use_bias=True, activation='relu', padding='same', name='conv2D_3x3')
    maxPool2D_2x2 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_first',
                                 name='maxPool2D_2x2')
    averagePool2D_2x2 = AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_first',
                                         name='averagePool2D_2x2')
    expand_dims = Lambda(lambda x: K.expand_dims(x, axis=1))
    MLP_pooled_1D_2n = MLPConv(num_outputs=(1,))
    MLP_pooled_1D_3n = MLPConv(num_outputs=(1,))
    MLP_pooled_2D_2x2 = MLPConv(num_outputs=(1,))
    MLP_pooled_2D_3x3 = MLPConv(num_outputs=(1,))
    MLP_pooled_lstms = MLPLSTM(num_outputs=(1,))


    def bi_lstms_network():
        layers = [Bidirectional(LSTM(units=50, activation='sigmoid', return_sequences=True,
                                     kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform',
                                     input_shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH)), merge_mode='concat'),
                  Bidirectional(LSTM(units=100, activation='sigmoid', return_sequences=True,
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

    pooled_1D_2n = maxPool2D_2x2(expand_dims(concat_conv1D_2n))
    pooled_1D_2n = Flatten()(pooled_1D_2n)
    pooled_1D_2n = MLP_pooled_1D_2n(pooled_1D_2n)

    pooled_1D_3n = maxPool2D_2x2(expand_dims(concat_conv1D_3n))
    pooled_1D_3n = Flatten()(pooled_1D_3n)
    pooled_1D_3n = MLP_pooled_1D_3n(pooled_1D_3n)

    pooled_2D_2x2 = maxPool2D_2x2(concat_2D_2x2)
    pooled_2D_2x2 = Flatten()(pooled_2D_2x2)
    pooled_2D_2x2 = MLP_pooled_2D_2x2(pooled_2D_2x2)

    pooled_2D_3x3 = maxPool2D_2x2(concat_2D_3x3)
    pooled_2D_3x3 = Flatten()(pooled_2D_3x3)
    pooled_2D_3x3 = MLP_pooled_2D_3x3(pooled_2D_3x3)

    pooled_lstms = averagePool2D_2x2(expand_dims(concat_bilstms))
    pooled_lstms = Flatten()(pooled_lstms)
    pooled_lstms = MLP_pooled_lstms(pooled_lstms)

    model = Model(inputs=[embedded_sentence_A, embedded_sentence_B], outputs=[pooled_1D_2n, pooled_1D_3n, pooled_2D_2x2, pooled_2D_3x3, pooled_lstms])

    return model

net = SemanticSimilarityNetwork()
net.summary()