from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input, concatenate, Bidirectional, LSTM, Reshape, Layer, \
    ReLU, Lambda, Conv1D, add, AveragePooling2D, BatchNormalization
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

selected_activation='tanh'

def tf_pearson(y_true, y_pred):
    return tf.contrib.metrics.streaming_pearson_correlation(y_pred, y_true)[1]

TRAIN = True

class MLPConv(Layer):
    def __init__(self, num_outputs):
        super(MLPConv, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        super(MLPConv, self).build(input_shape)
        self.dense_1 = Dense(100, activation=selected_activation, kernel_initializer='glorot_uniform', use_bias=True)
        self.dense_2 = Dense(50, activation=selected_activation, kernel_initializer='glorot_uniform', use_bias=True)
        self.dense_3 = Dense(25, activation=selected_activation, kernel_initializer='glorot_uniform', use_bias=True)
        self.dense_4 = Dense(1, activation=selected_activation, kernel_initializer='glorot_uniform', use_bias=True)

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
        self.dense_1 = Dense(1000, activation=selected_activation, kernel_initializer='glorot_uniform', use_bias=True)
        self.dense_2 = Dense(100, activation=selected_activation, kernel_initializer='glorot_uniform', use_bias=True)
        self.dense_3 = Dense(10, activation=selected_activation, kernel_initializer='glorot_uniform', use_bias=True)
        self.dense_4 = Dense(1, activation=selected_activation, kernel_initializer='glorot_uniform', use_bias=True)

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
        self.dense_1 = Dense(5, activation=selected_activation, kernel_initializer='glorot_uniform', use_bias=True)
        self.dense_2 = Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform', use_bias=True)

    def call(self, input):
        x = self.dense_1(input)
        x = self.dense_2(x)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)


def SemanticSimilarityNetwork():

    embedded_sentence_A = Input(shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH)) #(20, 300)
    embedded_sentence_B = Input(shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH)) #(20, 300)


    conv1D_2n = Conv1D(filters=10, kernel_size=2, kernel_initializer='glorot_uniform',
                       bias_initializer='glorot_uniform',input_shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH),
                       use_bias=True, activation=selected_activation, padding='same', name='conv1D_2n')
    conv1D_3n = Conv1D(filters=10, kernel_size=3, kernel_initializer='glorot_uniform',
                       bias_initializer='glorot_uniform',input_shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH),
                       use_bias=True, activation=selected_activation,padding='same', name='conv1D_3n')
    conv2D_2x2 = Conv2D(filters=10, kernel_size=(2, 2), kernel_initializer='glorot_uniform',
                        bias_initializer='glorot_uniform', input_shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH),
                        use_bias=True, activation=selected_activation,
                        padding='same', name='conv2D_2x2')
    conv2D_3x3 = Conv2D(filters=10, kernel_size=(3,3), kernel_initializer='glorot_uniform',
                        bias_initializer='glorot_uniform', input_shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH),
                        use_bias=True, activation=selected_activation, padding='same', name='conv2D_3x3')
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
    MLP_score = MLPScore(num_outputs=(1,))


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

    out =  Lambda(lambda x: K.stack(x, axis=-1))([pooled_1D_2n, pooled_1D_3n, pooled_2D_2x2, pooled_2D_3x3,
                                                   pooled_lstms])
    out = Flatten()(out)
    out = MLP_score(out)

    model = Model(inputs=[embedded_sentence_A, embedded_sentence_B], outputs=out)

    return model


network = SemanticSimilarityNetwork()
network.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error', metrics=['mean_absolute_error', tf_pearson])
network.summary()
model_head = 'model_semanticsimilaritynetwork'
time = time.strftime("%Y%m%d%H%M%S")
dataset_size = 5700
test_size = 1300
epochs = 50
model_name = model_head + '/' + configurations.EMBEDDER+str(EMBEDDING_LENGTH) + '_ep=' + str(epochs)+  '_act=' + selected_activation + '_slen=' + str(configurations.MAX_TEXT_WORD_LENGTH) \
             + '_dsize=' + str(dataset_size)+ '_tsize=' + str(test_size)+ '_' + time + '.h5'

sentences_A, sentences_B, scores = read_sts_data('test')
sentences_A = np.array([get_ready_vector(sentence) for sentence in sentences_A[0:test_size]])
sentences_B = np.array([get_ready_vector(sentence) for sentence in sentences_B[0:test_size]])
scores = minmax_scale(scores, feature_range=(0, 0.99))
scores = scores[0:test_size]

val_data = [[sentences_A, sentences_B], scores]



if TRAIN:
    #epoch_end_callback = LambdaCallback(on_epoch_end=epoch_test)
    reduce_lr = ReduceLROnPlateau(monitor='loss', patience=5, factor=0.1, verbose=1, min_lr=0.0001)
    #checkpoint_callback = ModelCheckpoint(
        #filepath='trained_models/model_independent_2_02_RegularRNN_Fasttext_mixedmargin_update00.h5', period=1)
    os.mkdir('./logs/'+ model_head + '/' + model_name)
    tensorboard = TensorBoard(log_dir='./logs/'+ model_head + '/' + model_name,
                              histogram_freq=0,
                              batch_size=BATCH_SIZE,
                              write_graph=True, write_grads=False, write_images=False, embeddings_freq=0,
                              embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None,
                              update_freq='epoch')
    data_generator = Native_DataGenerator_for_SemanticSimilarityNetwork(batch_size=BATCH_SIZE, dataset_size=dataset_size)
    K.get_session().run(tf.local_variables_initializer())
    network.fit_generator(generator=data_generator,validation_data=val_data,shuffle=True, epochs=epochs, workers=1, use_multiprocessing=False,
                        callbacks=[reduce_lr, tensorboard])
    os.mkdir('trained_models/' + model_head)
    network.save('./trained_models/' + model_name + '.h5')
else:
    #network = load_model('trained_models\model_independent_2_02_RegularRNN_Fasttext_mixedmargin.h5')
    network.summary()