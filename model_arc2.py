from keras.layers import Conv1D, Conv2D, BatchNormalization, MaxPooling2D, Dense, Reshape, Flatten, Input, concatenate, Lambda
from keras.models import Sequential, Model
from keras.optimizers import Adam
import tensorflow as tf
from config.configurations import ELMO_VECTOR_LENGTH, FASTTEXT_VECTOR_LENGTH, EMBEDDER
from data_utilities.generator import Native_DataGenerator_for_Arc2

if EMBEDDER == 'FASTTEXT':
    EMBEDDING_LENGTH = FASTTEXT_VECTOR_LENGTH
else:
    EMBEDDING_LENGTH = ELMO_VECTOR_LENGTH

COMBINATION_COUNT = 1944 #MAX_TEXT_WORD_LENGTH * 2 #1944
BATCH_SIZE = 112

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
    model.add(Dense(activation='softplus', units=1, use_bias=True))

    return model


pos_in = Input(shape=(COMBINATION_COUNT, EMBEDDING_LENGTH))
neg_in = Input(shape=(COMBINATION_COUNT, EMBEDDING_LENGTH))


net = create_network(input_shape=(None, EMBEDDING_LENGTH))

pos_out = net(pos_in)
neg_out = net(neg_in)
net_out = concatenate([pos_out, neg_out], axis=0)


model = Model(inputs=[pos_in, neg_in], outputs=net_out)
model.compile(optimizer=Adam(lr=0.001), loss=hinge_loss)


data_generator = Native_DataGenerator_for_Arc2(batch_size=BATCH_SIZE)

model.fit_generator(generator=data_generator, shuffle=True, epochs=10, workers=1, use_multiprocessing=False)
model.save('model_00.h5')


