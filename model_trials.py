from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input, concatenate, Bidirectional, LSTM, Reshape, Layer, \
    ReLU, Lambda, Conv1D, add, AveragePooling2D, TimeDistributed
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, LambdaCallback, EarlyStopping, ReduceLROnPlateau
import keras.backend as K
from config import configurations
from data_utilities.generator import Native_DataGenerator_for_SemanticSimilarityNetwork
import numpy as np
from texttovector import get_ready_vector, get_ready_vector_on_batch
import time
from data_utilities.datareader import read_sts_data
from sklearn.preprocessing import minmax_scale

DIM = 9
MAX_TEXT_WORD_LENGTH = configurations.MAX_TEXT_WORD_LENGTH #10
MAX_WORD_CHARACTER_LENGTH = configurations.MAX_WORD_CHARACTER_LENGTH #60
EMBEDDING_LENGTH = configurations.EMBEDDING_LENGTH #300
ALPHABET_LENGTH = configurations.ALPHABET_LENGTH #54
WORD_TO_WORD_COMBINATIONS = int(MAX_TEXT_WORD_LENGTH * MAX_TEXT_WORD_LENGTH) #10*10
WORD_TENSOR_DEPTH = int((ALPHABET_LENGTH * MAX_WORD_CHARACTER_LENGTH) / DIM) #(54*60/9 = 360)
BATCH_SIZE = configurations.BATCH_SIZE


TRAIN = True

class MLPConv(Layer):
    def __init__(self, num_outputs):
        super(MLPConv, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        super(MLPConv, self).build(input_shape)
        self.dense_1 = Dense(100, activation='relu', kernel_initializer='glorot_uniform', use_bias=True)
        self.dense_2 = Dense(50, activation='relu', kernel_initializer='glorot_uniform', use_bias=True)
        self.dense_3 = Dense(25, activation='relu', kernel_initializer='glorot_uniform', use_bias=True)
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


class MLPScore(Layer):
    def __init__(self, num_outputs):
        super(MLPScore, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        super(MLPScore, self).build(input_shape)
        self.dense_1 = Dense(5, activation='relu', kernel_initializer='glorot_uniform', use_bias=True)
        self.dense_2 = Dense(1, activation='relu', kernel_initializer='glorot_uniform', use_bias=True)

    def call(self, input):
        x = self.dense_1(input)
        x = self.dense_2(x)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)


def SemanticSimilarityNetwork():

    embedded_sentence_A = Input(shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH)) #(20, 300)
    embedded_sentence_B = Input(shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH)) #(20, 300)




    def bi_lstms_network():
        layers = [
                  Bidirectional(LSTM(units=300, activation='sigmoid', return_sequences=True,
                                     kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform',
                                     input_shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH)), merge_mode='concat'),
                  Bidirectional(LSTM(units=600, activation='sigmoid', return_sequences=True,
                                     kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform',
                                     input_shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH)), merge_mode='concat'),
                  TimeDistributed(Dense(600, activation='relu')),
                  TimeDistributed(Dense(300, activation='relu')),
                  TimeDistributed(Dense(100, activation='relu')),
                  TimeDistributed(Dense(30, activation='relu')),
                 ]

        def shared_layers(x):
            for layer in layers:
                x = layer(x)
            return x
        return shared_layers

    bi_lstms = bi_lstms_network()

    out_A = bi_lstms(embedded_sentence_A)
    out_B = bi_lstms(embedded_sentence_B)



    out = concatenate([out_A, out_B])
    out = Flatten()(out)
    out = Dense(1200, activation='relu')(out)
    out = Dense(600, activation='relu')(out)
    out = Dense(200, activation='relu')(out)
    out = Dense(100, activation='relu')(out)
    out = Dense(10, activation='relu')(out)
    out = Dense(1, activation='relu')(out)
    #out = MLP_score(out)

    model = Model(inputs=[embedded_sentence_A, embedded_sentence_B], outputs=out)

    return model


network = SemanticSimilarityNetwork()
network.compile(optimizer=Adam(lr=0.01), loss='mean_squared_error', metrics=['mean_absolute_error'])
network.summary()
model_name = 'model_semanticsimilaritynetwork'

sentences_A, sentences_B, scores = read_sts_data('test')
sentences_A = np.array([get_ready_vector(sentence) for sentence in sentences_A[0:100]])
sentences_B = np.array([get_ready_vector(sentence) for sentence in sentences_B[0:100]])

scores = scores[0:100]

val_data = [[sentences_A, sentences_B], scores]



if TRAIN:
    #epoch_end_callback = LambdaCallback(on_epoch_end=epoch_test)
    reduce_lr = ReduceLROnPlateau(monitor='loss', patience=5, factor=0.1, verbose=1, min_lr=0.000001)
    #checkpoint_callback = ModelCheckpoint(
        #filepath='trained_models/model_independent_2_02_RegularRNN_Fasttext_mixedmargin_update00.h5', period=1)
    tensorboard = TensorBoard(log_dir='./logs/'+ model_name, histogram_freq=0,
                              batch_size=BATCH_SIZE,
                              write_graph=True, write_grads=False, write_images=False, embeddings_freq=0,
                              embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None,
                              update_freq='batch')
    data_generator = Native_DataGenerator_for_SemanticSimilarityNetwork(batch_size=BATCH_SIZE)
    network.fit_generator(generator=data_generator,validation_data=val_data,shuffle=True, epochs=500, workers=1, use_multiprocessing=False,
                        callbacks=[reduce_lr, tensorboard])
    network.save('trained_models/' + model_name + '/' + model_name + '_' + time.strftime("%Y%m%d%H%M%S") + '.h5')
else:
    #network = load_model('trained_models\model_independent_2_02_RegularRNN_Fasttext_mixedmargin.h5')
    network.summary()