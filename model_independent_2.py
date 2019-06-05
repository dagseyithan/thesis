from keras.layers import Conv1D, Conv2D, BatchNormalization, MaxPooling2D, Dense, Reshape, Flatten, Input, concatenate, \
    Lambda, CuDNNGRU, AveragePooling3D, Bidirectional, Softmax, Multiply
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

TRAIN = True


def hinge_loss(y_true, y_pred, N=1000, beta = 1000, epsilon=1e-8):
    anchor = Lambda(lambda x: x[:, 0:N])(y_pred)
    positive = Lambda(lambda x: x[:, N:N * 2])(y_pred)
    negative = Lambda(lambda x: x[:, N * 2:N * 3])(y_pred)


    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

    pos_dist = -tf.log(-tf.divide((pos_dist), beta) + 1 + epsilon)
    neg_dist = -tf.log(-tf.divide((N - neg_dist), beta) + 1 + epsilon)

    loss = neg_dist + pos_dist

    return loss


input_a = Input(shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH))

conv2k = Conv1D(filters=400, kernel_size=2, kernel_initializer='glorot_uniform',
                  input_shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH), use_bias=True, activation='relu', padding='same')
conv3k = Conv1D(filters=400, kernel_size=3, kernel_initializer='glorot_uniform',
                  input_shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH), use_bias=True, activation='relu', padding='same')
conv2k_out = conv2k(input_a)
conv3k_out = conv3k(input_a)
concat_convs_a = concatenate([conv2k_out, conv3k_out], axis=1)

gru_concat = Bidirectional(CuDNNGRU(units=200, return_sequences=True, input_shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH)), merge_mode='concat')
gru_a = Bidirectional(CuDNNGRU(units=400, return_sequences=True, input_shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH)), merge_mode='concat')
gru_last = Bidirectional(CuDNNGRU(units=25, return_sequences=True, input_shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH)), merge_mode='concat')
gru_concat_out = gru_concat(concat_convs_a)
gru_a_out = gru_a(input_a)


def common_network():
    layers = [Conv2D(filters=100, kernel_size=(2, 2), kernel_initializer='glorot_uniform', data_format='channels_first',
                     use_bias=True,activation='relu', padding='same'),
              MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid',data_format='channels_first'),
              Conv2D(filters=100, kernel_size=(2, 2), kernel_initializer='glorot_uniform',
                     data_format='channels_first', use_bias=True,
                     activation='relu', padding='same'),
              MaxPooling2D(pool_size=(10, 10), strides=None, padding='valid', data_format='channels_first')
    ]

    def shared_layers(x):
        for layer in layers:
            x = layer(x)
        return x

    return shared_layers

common_net = common_network()
gru_concat_out = Reshape((2, MAX_TEXT_WORD_LENGTH, 400))(gru_concat_out)
gru_a_out = Reshape((2, MAX_TEXT_WORD_LENGTH, 400))(gru_a_out)
common_concat_out = common_net(gru_concat_out)
common_gru_out = common_net(gru_a_out)
common_common_out = concatenate([common_concat_out, common_gru_out], axis=1)
common_common_out = Reshape((MAX_TEXT_WORD_LENGTH, 200))(common_common_out)
gru_a_out = Reshape((MAX_TEXT_WORD_LENGTH, 800))(gru_a_out)

last_concat =  concatenate([common_common_out, gru_a_out], axis=-1)
gru_last_out = gru_last(last_concat)

x = Flatten()(gru_last_out)
x = Dense(units=1000, activation='relu')(x)
out = Dense(units=1000, activation='sigmoid')(x)

net = Model([input_a], out)
net.summary()
plot_model(net, 'model_independent2.png', show_shapes=True, show_layer_names=True)


anchor_in = Input(shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH))
pos_in = Input(shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH))
neg_in = Input(shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH))

anchor_out = net([anchor_in])
pos_out = net([pos_in])
neg_out = net([neg_in])

net_out = concatenate([anchor_out, pos_out, neg_out], axis=-1, name='net_out')

model = Model(inputs=[anchor_in, pos_in, neg_in], outputs=net_out)
model.compile(optimizer=Adam(lr=0.0001), loss=hinge_loss)
model.summary()

def get_similarity_from_independent_model_2(textA, textB):
    vec_A = model.predict_on_batch([np.reshape(get_ready_vector(textA), (1,MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH)),
                                   np.reshape(get_ready_vector(textA), (1,MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH)),
                                   np.reshape(get_ready_vector(textA), (1,MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH))])
    vec_B = model.predict_on_batch([np.reshape(get_ready_vector(textB), (1,MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH)),
                                   np.reshape(get_ready_vector(textB), (1,MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH)),
                                   np.reshape(get_ready_vector(textB), (1,MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH))])
    return cosine(vec_A, vec_B)


def epoch_test(epoch, logs):
    print(get_similarity_from_independent_model_2('Abdeckkappen Ø 5 mm weiß, 20 Stück', 'Abdeckkappe 5 mm weiss'))
    print(get_similarity_from_independent_model_2('Abdeckkappen Ø 5 mm weiß, 20 Stück', 'Abdeckkappe 6 mm weiss'))
    print('\n\n')

    test = 'Teppich MICHALSKY München anthrazit 133x190 cm'
    print(get_similarity_from_independent_model_2(test, 'Teppich MM München 133x190cm anthrazit'))
    print(get_similarity_from_independent_model_2('Teppich MM München 133x190cm anthrazit', test))
    print('\n')
    print(get_similarity_from_independent_model_2(test, 'Vliesfotot. 184x248 Brooklyn'))
    print(get_similarity_from_independent_model_2('Vliesfotot. 184x248 Brooklyn', test))
    print('\n')
    print(get_similarity_from_independent_model_2('Vliesfotot. 184x248 Brooklyn', 'Vliesfotot. 184x248 Brooklyn'))


epoch_end_callback = keras.callbacks.LambdaCallback(on_epoch_end=epoch_test)

if TRAIN:
    # model = load_model('trained_models/model_arc2_02_concat.h5', custom_objects={'hinge_loss': hinge_loss})
    data_generator = Native_DataGenerator_for_IndependentModel(batch_size=BATCH_SIZE)
    model.fit_generator(generator=data_generator, shuffle=True, epochs=100, workers=1, use_multiprocessing=False, callbacks=[epoch_end_callback])
    model.save('trained_models/model_independent_06_BiGRU_Fasttext_hardmargin_differentarchitectures.h5')
else:
    model = load_model('trained_models/model_independent_02_BiGRU_FastText_hardmargin.h5', custom_objects={'hinge_loss': hinge_loss})
    model.summary()




print('\n')

print(get_similarity_from_independent_model('Unterkonstruktion tepro Pent Roof 6x4 193 x 112 cm, silber', 'Unterk.Pent Roof 6x4für 7116/7209/7236'))
print(get_similarity_from_independent_model('Unterkonstruktion tepro Pent Roof 6x4 193 x 112 cm, silber', 'Unterkonstruktion tepro Riverton 6x4 192,2 x 112,1 cm, silber'))
print(get_similarity_from_independent_model('Unterkonstruktion tepro Pent Roof 6x4 193 x 112 cm, silber', 'Feuchtraumkabel NYM-J 3G1,5 10M'))
print(get_similarity_from_independent_model('Unterkonstruktion tepro Pent Roof 6x4 193 x 112 cm, silber', 'Unterk.Riverton 6x4 HAUSTYP1/7124/7240'))
print(get_similarity_from_independent_model('Unterkonstruktion tepro Riverton 6x4 192,2 x 112,1 cm, silber', 'Unterk.Pent Roof 6x4für 7116/7209/7236'))
print(get_similarity_from_independent_model('Unterkonstruktion tepro Riverton 6x4 192,2 x 112,1 cm, silber', 'Unterk.Riverton 6x4 HAUSTYP1/7124/7240'))
print(get_similarity_from_independent_model('Unterk.Riverton 6x4 HAUSTYP1/7124/7240', 'Unterkonstruktion tepro Pent Roof 6x4 193 x 112 cm, silber'))
print(get_similarity_from_independent_model('Unterk.Riverton 6x4 HAUSTYP1/7124/7240', 'Unterkonstruktion tepro Riverton 6x4 192,2 x 112,1 cm, silber'))
