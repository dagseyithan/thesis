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
import time

DIM = 9
MAX_TEXT_WORD_LENGTH = configurations.MAX_TEXT_WORD_LENGTH #10
MAX_WORD_CHARACTER_LENGTH = configurations.MAX_WORD_CHARACTER_LENGTH #60
EMBEDDING_LENGTH = configurations.FASTTEXT_VECTOR_LENGTH #300
ALPHABET_LENGTH = configurations.ALPHABET_LENGTH #54
WORD_TO_WORD_COMBINATIONS = int(MAX_TEXT_WORD_LENGTH * MAX_TEXT_WORD_LENGTH) #10*10
WORD_TENSOR_DEPTH = int((ALPHABET_LENGTH * MAX_WORD_CHARACTER_LENGTH) / DIM) #(54*60/9 = 360)
BATCH_SIZE = configurations.BATCH_SIZE

class MatrixMean(Layer):
    def __init__(self, num_outputs):
        super(MatrixMean, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        super(MatrixMean, self).build(input_shape)

    def call(self, input):
        matrices_AB = input[0]
        matrices_AB_r = input[1]
        out = []
        for i in range(BATCH_SIZE):
            out.append(add([matrices_AB[i,:,:], matrices_AB_r[i,:,:]]) * 0.5)
        out = K.stack(out, axis=0)
        out = K.reshape(out, (BATCH_SIZE, MAX_TEXT_WORD_LENGTH, MAX_TEXT_WORD_LENGTH))
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], MAX_TEXT_WORD_LENGTH, MAX_TEXT_WORD_LENGTH)


def UnificationNetwork():

    embedded_sentence_A = Input(shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH)) #(10, 300)
    embedded_sentence_B = Input(shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH)) #(10, 300)
    word_tensors_A =  Input(shape=(WORD_TENSOR_DEPTH * WORD_TO_WORD_COMBINATIONS, DIM)) #(36000, 9)
    word_tensors_B = Input(shape=(WORD_TENSOR_DEPTH * WORD_TO_WORD_COMBINATIONS, DIM)) #(36000, 9)
    word_tensors_A_r = Input(shape=(WORD_TENSOR_DEPTH * WORD_TO_WORD_COMBINATIONS, DIM)) #(36000, 9)
    word_tensors_B_r = Input(shape=(WORD_TENSOR_DEPTH * WORD_TO_WORD_COMBINATIONS, DIM)) #(36000, 9)
    tensor_masks = Input(shape=(WORD_TO_WORD_COMBINATIONS, WORD_TENSOR_DEPTH)) #(100, 360)
    tensor_masks_r = Input(shape=(WORD_TO_WORD_COMBINATIONS, WORD_TENSOR_DEPTH)) #(100, 360)


    def SemanticSimilarityNetwork():
        layers = [
                  Conv1D(filters=400, kernel_size=2, kernel_initializer='glorot_uniform',
                         input_shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH), use_bias=True, activation='relu',
                         padding='same'),
                  Conv1D(filters=400, kernel_size=3, kernel_initializer='glorot_uniform',
                         input_shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH), use_bias=True, activation='relu',
                         padding='same'),
                  Conv1D(filters=400, kernel_size=2, kernel_initializer='glorot_uniform',
                         input_shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH), use_bias=True, activation='relu',
                         padding='same'),
                  Conv1D(filters=400, kernel_size=3, kernel_initializer='glorot_uniform',
                         input_shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH), use_bias=True, activation='relu',
                         padding='same'),
                  Bidirectional(LSTM(units=200, activation='sigmoid', return_sequences=True,
                                     input_shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH)), merge_mode='concat'),
                  Bidirectional(LSTM(units=200, activation='sigmoid', return_sequences=True,
                                     input_shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH)), merge_mode='concat'),
                  ]

        def shared_layers(x):
            out = []
            for layer in layers:
                x = layer(x)
                out.append(x)
            return x
        return shared_layers

    semantic_similarity_network = SemanticSimilarityNetwork()
    structural_similarity_network = StructuralSimilarityNetwork()
    matrix_mean = MatrixMean((BATCH_SIZE, MAX_TEXT_WORD_LENGTH, MAX_TEXT_WORD_LENGTH))


    embedded_sentence_A_out = semantic_similarity_network(embedded_sentence_A)
    embedded_sentence_B_out = semantic_similarity_network(embedded_sentence_B)

    word_comparison_AB = structural_similarity_network([word_tensors_A, word_tensors_B, tensor_masks])
    word_comparison_AB_r = structural_similarity_network([word_tensors_A_r, word_tensors_B_r, tensor_masks_r])

    structural_similarity_combination_matrices = matrix_mean([word_comparison_AB, word_comparison_AB_r])

    out = concatenate([embedded_sentence_A_out, embedded_sentence_B_out, structural_similarity_combination_matrices])


    model = Model(inputs=[embedded_sentence_A, embedded_sentence_B, word_tensors_A, word_tensors_B, word_tensors_A_r,
                          word_tensors_B_r, tensor_masks, tensor_masks_r], outputs=out)

    return model


unification_network = UnificationNetwork()
unification_network.summary()
