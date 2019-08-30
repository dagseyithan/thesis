from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input, concatenate, Bidirectional, LSTM, Reshape, Layer, \
    ReLU, Lambda, Conv1D, add
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, LambdaCallback
import keras.backend as K
from config import configurations
import numpy as np
from encoder import encode_word, convert_to_tensor
from structural_similarity_network import StructuralSimilarityNetwork
from semantic_similarity_network_uni import SemanticSimilarityNetwork_Uni
import time

DIM = 9
MAX_TEXT_WORD_LENGTH = configurations.MAX_TEXT_WORD_LENGTH
MAX_WORD_CHARACTER_LENGTH = configurations.MAX_WORD_CHARACTER_LENGTH
EMBEDDING_LENGTH = configurations.EMBEDDING_LENGTH
ALPHABET_LENGTH = configurations.ALPHABET_LENGTH
WORD_TO_WORD_COMBINATIONS = int(MAX_TEXT_WORD_LENGTH * MAX_TEXT_WORD_LENGTH)
WORD_TENSOR_DEPTH = int((ALPHABET_LENGTH * MAX_WORD_CHARACTER_LENGTH) / DIM)
BATCH_SIZE = configurations.BATCH_SIZE


class MLP(Layer):
    def __init__(self, num_outputs):
        super(MLP, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        super(MLP, self).build(input_shape)
        self.dense_1 = Dense(100, activation='sigmoid', kernel_initializer='glorot_uniform', use_bias=True)
        self.dense_2 = Dense(50, activation='sigmoid', kernel_initializer='glorot_uniform', use_bias=True)
        self.dense_3 = Dense(25, activation='sigmoid', kernel_initializer='glorot_uniform', use_bias=True)
        self.dense_4 = Dense(5, activation='sigmoid', kernel_initializer='glorot_uniform', use_bias=True)
        self.dense_5 = Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform', use_bias=True)

    def call(self, input):
        x = self.dense_1(input)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.dense_4(x)
        x = self.dense_5(x)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)


def UnificationNetwork():

    embedded_sentence_A = Input(shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH))
    embedded_sentence_B = Input(shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH))
    word_tensors_A =  Input(shape=(WORD_TENSOR_DEPTH * WORD_TO_WORD_COMBINATIONS, DIM))
    word_tensors_B = Input(shape=(WORD_TENSOR_DEPTH * WORD_TO_WORD_COMBINATIONS, DIM))
    word_tensors_A_r = Input(shape=(WORD_TENSOR_DEPTH * WORD_TO_WORD_COMBINATIONS, DIM))
    word_tensors_B_r = Input(shape=(WORD_TENSOR_DEPTH * WORD_TO_WORD_COMBINATIONS, DIM))
    tensor_masks = Input(shape=(WORD_TO_WORD_COMBINATIONS, WORD_TENSOR_DEPTH))
    tensor_masks_r = Input(shape=(WORD_TO_WORD_COMBINATIONS, WORD_TENSOR_DEPTH))



    semantic_similarity_network = SemanticSimilarityNetwork_Uni()
    structural_similarity_network = StructuralSimilarityNetwork()
    structural_similarity_network.trainable = False
    structural_LSTM = LSTM(units=30, kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform',
                           return_sequences=True)
    scoring_perceptron = MLP((BATCH_SIZE, 1))


    semantic_similarity_out = semantic_similarity_network([embedded_sentence_A, embedded_sentence_B])
    structural_similarity_out = structural_similarity_network([word_tensors_A, word_tensors_A_r, word_tensors_B,
                                                               word_tensors_B_r, tensor_masks, tensor_masks_r])


    structural_similarity_out = structural_LSTM(structural_similarity_out)
    structural_similarity_out = Flatten()(structural_similarity_out)

    concat = [semantic_similarity_out, structural_similarity_out]
    out = scoring_perceptron(concat)




    model = Model(inputs=[embedded_sentence_A, embedded_sentence_B, word_tensors_A, word_tensors_B, word_tensors_A_r,
                          word_tensors_B_r, tensor_masks, tensor_masks_r], outputs=out)

    return model


unification_network = UnificationNetwork()
unification_network.summary()
