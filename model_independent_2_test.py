from keras.layers import Conv1D, Conv2D, MaxPooling2D, Dense, Reshape, Flatten, Input, concatenate, \
    Lambda, CuDNNGRU, Bidirectional, BatchNormalization, CuDNNLSTM
from keras.models import Sequential, Model, load_model
import keras.callbacks
from keras.optimizers import Adam, Adadelta
import keras.backend as K
import tensorflow as tf
from config.configurations import MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH, BATCH_SIZE
from data_utilities.generator import Native_DataGenerator_for_IndependentModel
from keras.utils import plot_model
from texttovector import get_ready_vector
from scipy.spatial.distance import cosine
import numpy as np

def hinge_loss(y_true, y_pred, N=1000, beta=1000.0, epsilon=1e-5):
    anchor = Lambda(lambda x: x[:, 0:N])(y_pred)
    positive = Lambda(lambda x: x[:, N:N * 2])(y_pred)
    negative = Lambda(lambda x: x[:, N * 2:N * 3])(y_pred)

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

    pos_dist = -tf.log(-tf.divide((pos_dist), beta) + 1 + epsilon)
    neg_dist = -tf.log(-tf.divide((N - neg_dist), beta) + 1 + epsilon)

    loss = neg_dist + pos_dist


    return loss

model = load_model('trained_models/model_independent_2_01_Fasttext_mixedmargin.h5', custom_objects={'hinge_loss': hinge_loss})

def get_similarity_from_independent_model_2(textA, textB):
    vec_A = model.predict_on_batch([np.reshape(get_ready_vector(textA), (1,MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH)),
                                   np.reshape(get_ready_vector(textA), (1,MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH)),
                                   np.reshape(get_ready_vector(textA), (1,MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH))])
    vec_B = model.predict_on_batch([np.reshape(get_ready_vector(textB), (1,MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH)),
                                   np.reshape(get_ready_vector(textB), (1,MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH)),
                                   np.reshape(get_ready_vector(textB), (1,MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH))])
    return cosine(vec_A, vec_B)



print(get_similarity_from_independent_model_2('Abdeckrahmen 1-fach Kopp Paris arktisweiß',
                                            'Abdeckrahmen 1-fach Paris arktis weiss'))

print(get_similarity_from_independent_model_2('Abdeckrahmen 1-fach Kopp Paris arktisweiß',
                                            'Abdeckrahmen Paris 1-fach silbern'))
