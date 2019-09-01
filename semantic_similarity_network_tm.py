from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input, concatenate, Bidirectional, LSTM, Reshape, Layer, \
    ReLU, Lambda, Conv1D, add, AveragePooling2D, BatchNormalization
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, LambdaCallback, EarlyStopping, ReduceLROnPlateau
import keras.backend as K
from config import configurations
from data_utilities.generator import Native_DataGenerator_for_SemanticSimilarityNetwork_TM, \
    Native_ValidationDataGenerator_for_SemanticSimilarityNetwork_TM

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


def triplet_loss(y_true, y_pred, beta = 300, N = 300, epsilon = configurations.EPSILON):
    anchor = Lambda(lambda x: x[0:BATCH_SIZE,:], output_shape=(BATCH_SIZE,1))(y_pred)
    positive = Lambda(lambda x: x[BATCH_SIZE:BATCH_SIZE*2,:], output_shape=(BATCH_SIZE,1))(y_pred)
    negative = Lambda(lambda x: x[BATCH_SIZE*2:BATCH_SIZE*3,:], output_shape=(BATCH_SIZE,1))(y_pred)

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)


    pos_dist = -tf.log(-tf.divide((pos_dist), beta) + 1 + epsilon)
    neg_dist = -tf.log(-tf.divide((N - neg_dist), beta) + 1 + epsilon)

    loss = neg_dist + pos_dist

    return loss

def tf_pearson(y_true, y_pred):
    return tf.contrib.metrics.streaming_pearson_correlation(y_pred, y_true)[1]

TRAIN = True


class MLPProject(Layer):
    def __init__(self, num_outputs):
        super(MLPProject, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        super(MLPProject, self).build(input_shape)
        self.dense_1 = Dense(900, activation=selected_activation, kernel_initializer='glorot_uniform', use_bias=True)
        self.dense_2 = Dense(600, activation=selected_activation, kernel_initializer='glorot_uniform', use_bias=True)
        self.dense_3 = Dense(300, activation='sigmoid', kernel_initializer='glorot_uniform', use_bias=True)

    def call(self, input):
        x = self.dense_1(input)
        x = self.dense_2(x)
        x = self.dense_3(x)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 300)



embedded_sentence = Input(shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH))

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
MLP_projector = MLPProject(num_outputs=(300,))


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

conv1D_2n_out_A = conv1D_2n(embedded_sentence)
conv1D_3n_out_A = conv1D_3n(embedded_sentence)
conv2D_2x2_out_A = conv2D_2x2(expand_dims(embedded_sentence))
conv2D_3x3_out_A = conv2D_3x3(expand_dims(embedded_sentence))
bi_lstms_out_A = bi_lstms(embedded_sentence)


pooled_1D_2n = Flatten()(maxPool2D_2x2(expand_dims(conv1D_2n_out_A)))
pooled_1D_3n = Flatten()(maxPool2D_2x2(expand_dims(conv1D_3n_out_A)))
pooled_2D_2x2 = Flatten()(maxPool2D_2x2(conv2D_2x2_out_A))
pooled_2D_3x3 = Flatten()(maxPool2D_2x2(conv2D_3x3_out_A))
pooled_lstms = Flatten()(averagePool2D_2x2(expand_dims(bi_lstms_out_A)))


out =  Lambda(lambda x: K.concatenate(x, axis=-1))([pooled_1D_2n, pooled_1D_3n, pooled_2D_2x2, pooled_2D_3x3,
                                               pooled_lstms])

out = MLP_projector(out)

embedder = Model(inputs=[embedded_sentence], outputs=out)


anchor_in = Input(shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH))
pos_in = Input(shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH))
neg_in = Input(shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH))



anchor_out = embedder(anchor_in)
pos_out = embedder(pos_in)
neg_out = embedder(neg_in)

net_out = concatenate([anchor_out, pos_out, neg_out], axis=0)

combiner_model = Model(inputs=[anchor_in, pos_in, neg_in], outputs=net_out)



network = combiner_model
network.compile(optimizer=Adam(lr=0.01), loss=triplet_loss)
network.summary()
model_head = 'model_semanticsimilarity_TM'
time = time.strftime("%Y%m%d%H%M%S")
dset = 'tarent'
dataset_size = 48000
test_size = 1000
epochs = 500
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
    data_generator = Native_DataGenerator_for_SemanticSimilarityNetwork_TM(batch_size=BATCH_SIZE)
    validation_generator = Native_ValidationDataGenerator_for_SemanticSimilarityNetwork_TM(batch_size=BATCH_SIZE)
    K.get_session().run(tf.local_variables_initializer())
    network.fit_generator(generator=data_generator,shuffle=True, validation_data=validation_generator,
                          epochs=epochs, workers=2, use_multiprocessing=False,
                        callbacks=[reduce_lr, tensorboard])
    if not os.path.exists('./trained_models/' + model_head):
        os.mkdir('./trained_models/' + model_head)
    embedder.save('./trained_models/' + 'tm_ebedder' + '.h5')
    network.save('./trained_models/' + model_name + '.h5')
else:
    #network = load_model('trained_models\model_independent_2_02_RegularRNN_Fasttext_mixedmargin.h5')
    network.summary()