from keras.layers import Conv1D, Conv2D, BatchNormalization, MaxPooling2D, Dense, Reshape, Flatten, Input, concatenate, Lambda
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
import tensorflow as tf
from config.configurations import ELMO_VECTOR_LENGTH, FASTTEXT_VECTOR_LENGTH, EMBEDDER, MAX_TEXT_WORD_LENGTH
from data_utilities.generator import Native_DataGenerator_for_Arc2, get_combinations
from texttovector import get_ready_vector
import numpy as np

if EMBEDDER == 'FASTTEXT':
    EMBEDDING_LENGTH = FASTTEXT_VECTOR_LENGTH
else:
    EMBEDDING_LENGTH = ELMO_VECTOR_LENGTH

COMBINATION_COUNT = 1944 #MAX_TEXT_WORD_LENGTH * 2 #1944
BATCH_SIZE = 112

TRAIN = True

def hinge_loss(y_true, y_pred, alpha = 1.0):

    slice_pos = lambda x: x[0:BATCH_SIZE,:]
    slice_neg = lambda x: x[BATCH_SIZE:BATCH_SIZE*2,:]

    positive = Lambda(slice_pos, output_shape=(BATCH_SIZE,1))(y_pred)
    negative = Lambda(slice_neg, output_shape=(BATCH_SIZE,1))(y_pred)

    #positive = K.reshape(positive, (BATCH_SIZE, ))
    #negative = K.reshape(negative, (BATCH_SIZE, ))

    basic_loss = alpha + negative - positive

    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0), axis=1)

    return loss

def create_network(input_shape):

    model = Sequential()
    model.add(BatchNormalization(input_shape = input_shape))
    model.add(Conv1D(filters=100, kernel_size=3, kernel_initializer='truncated_normal', input_shape=(None, EMBEDDING_LENGTH), use_bias=True, activation='relu', padding='same'))
    model.add(Reshape((COMBINATION_COUNT, 10, 10)))
    model.add(Conv2D(filters=40, kernel_size=(3, 3), kernel_initializer='truncated_normal', input_shape=(None, EMBEDDING_LENGTH), data_format='channels_first', use_bias=True, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_first'))
    model.add(Conv2D(filters=20, kernel_size=(3, 3), kernel_initializer='truncated_normal', input_shape=(None, EMBEDDING_LENGTH), data_format='channels_first', use_bias=True, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_first'))
    #model.add(Conv2D(filters=100, kernel_size=(3, 3), kernel_initializer='truncated_normal', input_shape=(None, EMBEDDING_LENGTH), data_format='channels_first', use_bias=True, activation='relu', padding='same'))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_first'))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(activation='relu', units=64, use_bias=True))
    model.add(Dense(activation='relu', units=32, use_bias=True))
    model.add(Dense(activation='sigmoid', units=1, use_bias=True))

    return model

if TRAIN:
    pos_in = Input(shape=(COMBINATION_COUNT, EMBEDDING_LENGTH))
    neg_in = Input(shape=(COMBINATION_COUNT, EMBEDDING_LENGTH))


    net = create_network(input_shape=(None, EMBEDDING_LENGTH))

    pos_out = net(pos_in)
    neg_out = net(neg_in)
    net_out = concatenate([pos_out, neg_out], axis=0)


    model = Model(inputs=[pos_in, neg_in], outputs=net_out)
    model.compile(optimizer=Adam(lr=0.0001), loss=hinge_loss)


    data_generator = Native_DataGenerator_for_Arc2(batch_size=BATCH_SIZE)

    model = load_model('trained_models/model_arc2_01_sigmoid.h5', custom_objects={'hinge_loss': hinge_loss})
    model.fit_generator(generator=data_generator, shuffle=True, epochs=10, workers=16, use_multiprocessing=True)
    model.save('model_arc2_01_sigmoid.h5')
else:
    model = load_model('trained_models/model_arc2_00.h5', custom_objects={'hinge_loss': hinge_loss})
    model.summary()
    print(len(model.layers))
    #model.layers.pop(3)
    #model.summary()
    #print(len(model.layers))
    combined_vector = np.reshape(get_combinations(get_ready_vector('Grünes Taschentuch'), get_ready_vector('Blaues Taschentuch'),
                                                max_text_length=MAX_TEXT_WORD_LENGTH,
                                                word_embedding_length=EMBEDDING_LENGTH), (1, COMBINATION_COUNT, EMBEDDING_LENGTH))

    combined_vector2 = np.reshape(get_combinations(get_ready_vector('Grünes Taschentuch'), get_ready_vector('Die Schule Schuhe'),
                                                max_text_length=MAX_TEXT_WORD_LENGTH,
                                                word_embedding_length=EMBEDDING_LENGTH), (1, COMBINATION_COUNT, EMBEDDING_LENGTH))

    combined_vector3 = np.reshape(get_combinations(get_ready_vector('Grünes Taschentuch'), get_ready_vector('Grünes Taschentuch'),
                                                max_text_length=MAX_TEXT_WORD_LENGTH,
                                                word_embedding_length=EMBEDDING_LENGTH), (1, COMBINATION_COUNT, EMBEDDING_LENGTH))
    print(combined_vector.shape)
    print(model.predict_on_batch([combined_vector, combined_vector]))
    print(model.predict_on_batch([combined_vector2, combined_vector2]))
    print(model.predict_on_batch([combined_vector3, combined_vector3]))



