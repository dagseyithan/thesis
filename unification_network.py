from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input, concatenate, Bidirectional, LSTM, Reshape, Layer, \
    ReLU, Lambda, Conv1D, add, AveragePooling2D, BatchNormalization
from keras.models import Model, load_model
from keras.utils import multi_gpu_model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, LambdaCallback, EarlyStopping, ReduceLROnPlateau
import keras.backend as K
from config import configurations
from data_utilities.generator import Native_DataGenerator_for_UnificationNetwork_SICK, \
    Native_ValidationDataGenerator_for_UnificationNetwork_SICK, Native_DataGenerator_for_UnificationNetwork_MSR, \
    Native_ValidationDataGenerator_for_UnificationNetwork_MSR, Native_DataGenerator_for_UnificationNetwork_STS, \
    Native_ValidationDataGenerator_for_UnificationNetwork_STS
import numpy as np
from texttovector import get_ready_vector
import time
from data_utilities.datareader import read_sts_data, read_sick_data
from sklearn.preprocessing import minmax_scale
import os
from structural_similarity_network import StructuralSimilarityNetwork
from semantic_similarity_network_uni import SemanticSimilarityNetwork_Uni
import time
import tensorflow as tf

DIM = 9
MAX_TEXT_WORD_LENGTH = configurations.MAX_TEXT_WORD_LENGTH
MAX_WORD_CHARACTER_LENGTH = configurations.MAX_WORD_CHARACTER_LENGTH
EMBEDDING_LENGTH = configurations.EMBEDDING_LENGTH
ALPHABET_LENGTH = configurations.ALPHABET_LENGTH
WORD_TO_WORD_COMBINATIONS = int(MAX_TEXT_WORD_LENGTH * MAX_TEXT_WORD_LENGTH)
WORD_TENSOR_DEPTH = int((ALPHABET_LENGTH * MAX_WORD_CHARACTER_LENGTH) / DIM)
BATCH_SIZE = configurations.BATCH_SIZE


def tf_pearson(y_true, y_pred):
    return tf.contrib.metrics.streaming_pearson_correlation(y_pred, y_true)[1]

TRAIN = True
selected_activation='relu'

class MLP(Layer):
    def __init__(self, num_outputs):
        super(MLP, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        super(MLP, self).build(input_shape)
        self.dense_1 = Dense(6300, activation=selected_activation, kernel_initializer='glorot_uniform', use_bias=True)
        self.dense_2 = Dense(630, activation=selected_activation, kernel_initializer='glorot_uniform', use_bias=True)
        self.dense_3 = Dense(63, activation=selected_activation, kernel_initializer='glorot_uniform', use_bias=True)
        self.dense_4 = Dense(21, activation=selected_activation, kernel_initializer='glorot_uniform', use_bias=True)
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

    with(tf.device('/gpu:0')):
        semantic_similarity_out = semantic_similarity_network([embedded_sentence_A, embedded_sentence_B])
    with(tf.device('/gpu:1')):
        structural_similarity_out = structural_similarity_network([word_tensors_A, word_tensors_A_r, word_tensors_B,
                                                               word_tensors_B_r, tensor_masks, tensor_masks_r])


    structural_similarity_out = structural_LSTM(structural_similarity_out)
    structural_similarity_out = Flatten()(structural_similarity_out)

    concat = concatenate([semantic_similarity_out, structural_similarity_out])
    out = scoring_perceptron(concat)




    model = Model(inputs=[embedded_sentence_A, embedded_sentence_B, word_tensors_A, word_tensors_B, word_tensors_A_r,
                          word_tensors_B_r, tensor_masks, tensor_masks_r], outputs=out)

    return model


network = UnificationNetwork()
network.summary()

network.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error', metrics=['mean_abdsolute_error', tf_pearson])
network.summary()
model_head = 'model_unificationnetwork'
time = time.strftime("%Y%m%d%H%M%S")
dset = 'STS'
dataset_size = 5700
test_size = 2300
epochs = 50
model_name = model_head + '/' + configurations.EMBEDDER+str(EMBEDDING_LENGTH) + '_dset=' + dset + '_bsize=' + str(BATCH_SIZE) +'_ep=' + str(epochs)+  '_act=' + selected_activation + '_slen=' + str(configurations.MAX_TEXT_WORD_LENGTH) \
             + '_dsize=' + str(dataset_size)+ '_tsize=' + str(test_size)+ '_' + time + '.h5'




if TRAIN:

    reduce_lr = ReduceLROnPlateau(monitor='loss', patience=5, factor=0.1, verbose=1, min_lr=0.0001)

    if not os.path.exists('./logs/'+  model_head):
        os.mkdir('./logs/'+  model_head)
    tensorboard = TensorBoard(log_dir='./logs/' + model_name,
                              histogram_freq=0,
                              batch_size=BATCH_SIZE,
                              write_graph=True, write_grads=False, write_images=False, embeddings_freq=0,
                              embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None,
                              update_freq='epoch')
    data_generator = Native_DataGenerator_for_UnificationNetwork_STS(batch_size=BATCH_SIZE)
    validation_generator = Native_ValidationDataGenerator_for_UnificationNetwork_STS(batch_size=BATCH_SIZE)
    K.get_session().run(tf.local_variables_initializer())
    network.fit_generator(generator=data_generator,validation_data=validation_generator,shuffle=True,
                          epochs=epochs, workers=32, use_multiprocessing=True,
                        callbacks=[reduce_lr, tensorboard])
    if not os.path.exists('./trained_models/' + model_head):
        os.mkdir('./trained_models/' + model_head)
    network.save('./trained_models/' + model_name + '.h5')
else:
    #network = load_model('trained_models\model_independent_2_02_RegularRNN_Fasttext_mixedmargin.h5')
    network.summary()