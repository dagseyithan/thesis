from keras.layers import Conv1D, Conv2D, BatchNormalization, MaxPooling2D, Dense, Reshape, Flatten, Input, concatenate, \
    Lambda, CuDNNGRU, AveragePooling3D, Bidirectional
from keras.models import Sequential, Model, load_model
import keras.callbacks
from keras.optimizers import Adam, Adadelta
import keras.backend as K
import tensorflow as tf
from config.configurations import MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH, BATCH_SIZE
from data_utilities.generator import Native_DataGenerator_for_IndependentModel
from keras.utils import plot_model
from texttovector import get_ready_vector
import numpy as np

TRAIN = True


def hinge_loss(y_true, y_pred, alpha=1.0, N=3.0, beta=3.0, epsilon=1e-8):

    slice_pos = lambda x: x[0:BATCH_SIZE, :]
    slice_neg = lambda x: x[BATCH_SIZE:BATCH_SIZE * 2, :]

    anchor_pos = Lambda(lambda x: x[0:BATCH_SIZE, :], output_shape=(BATCH_SIZE, 1))(y_pred)
    pos_anchor = Lambda(lambda x: x[BATCH_SIZE:BATCH_SIZE * 2, :], output_shape=(BATCH_SIZE, 1))(y_pred)
    anchor_neg = Lambda(lambda x: x[BATCH_SIZE * 2:BATCH_SIZE * 3, :], output_shape=(BATCH_SIZE, 1))(y_pred)
    neg_anchor = Lambda(lambda x: x[BATCH_SIZE * 3:BATCH_SIZE * 4, :], output_shape=(BATCH_SIZE, 1))(y_pred)

    basic_loss = alpha + anchor_neg - anchor_pos
    second_loss = alpha + neg_anchor - pos_anchor
    #simmetry_loss = K.sum(tf.abs(anchor_neg - neg_anchor), axis=-1) + K.sum(tf.abs(anchor_pos - pos_anchor), axis=-1)

    loss = K.sum(K.maximum(basic_loss, 0.0), axis=-1) + K.sum(K.maximum(second_loss, 0.0), axis=-1)

    return loss


input_a = Input(shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH))
input_b = Input(shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH))
conv_out = Conv1D(filters=400, kernel_size=3, kernel_initializer='glorot_uniform',
                  input_shape=(None, EMBEDDING_LENGTH), use_bias=True, activation='softplus', padding='same')
conv_out_a = conv_out(input_a)
conv_out_b = conv_out(input_b)
gru_a = Bidirectional(CuDNNGRU(units=100, return_sequences=True, input_shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH)), merge_mode='concat')
gru_b = Bidirectional(CuDNNGRU(units=100, return_sequences=True, input_shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH)), merge_mode='concat')
gru_a_out_a = gru_a(conv_out_a)
gru_a_out_b = gru_a(conv_out_b)
gru_b_out_a = gru_b(conv_out_a)
gru_b_out_b = gru_b(conv_out_b)
concat_a = concatenate([gru_a_out_a, gru_b_out_a], axis=1)
concat_b = concatenate([gru_a_out_b, gru_b_out_b], axis=1)
concat_a = Reshape((2, MAX_TEXT_WORD_LENGTH, 200))(concat_a)
concat_b = Reshape((2, MAX_TEXT_WORD_LENGTH, 200))(concat_b)
#avgpool3ded_a = AveragePooling3D(pool_size=(2, 1, 1), strides=None, padding='valid',
                                 #input_shape=(1, 2, MAX_TEXT_WORD_LENGTH, 100), data_format='channels_first')(concat_a)
#avgpool3ded_b = AveragePooling3D(pool_size=(2, 1, 1), strides=None, padding='valid',
                                 #input_shape=(1, 2, MAX_TEXT_WORD_LENGTH, 100), data_format='channels_first')(concat_b)

#concat_a = K.squeeze(concat_a, axis=1)
#concat_b = K.squeeze(concat_b, axis=1)


def common_network():
    layers = [Conv2D(filters=100, kernel_size=(3, 3), kernel_initializer='glorot_uniform',
                     input_shape=(2, MAX_TEXT_WORD_LENGTH, 200), data_format='channels_first',
                     use_bias=True,activation='softplus', padding='same'),
              MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid',data_format='channels_first'),
              Conv2D(filters=100, kernel_size=(3, 3), kernel_initializer='glorot_uniform',
                     data_format='channels_first', use_bias=True,
                     activation='softplus', padding='same'),
              MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_first')
    ]

    def shared_layers(x):
        for layer in layers:
            x = layer(x)
        return x

    return shared_layers

common_net = common_network()
a = common_net(concat_a)
b = common_net(concat_b)

concat_ab = concatenate([a, b], axis=-1)

x = Flatten()(concat_ab)
#x = BatchNormalization()(x)
x = Dense(activation='softplus', units=5000, use_bias=True)(x)
x = Dense(activation='softplus', units=2500, use_bias=True)(x)
x = Dense(activation='softplus', units=1000, use_bias=True)(x)
x = Dense(activation='softplus', units=100, use_bias=True)(x)
x = Dense(activation='softplus', units=10, use_bias=True)(x)
out = Dense(activation='softplus', units=1, use_bias=True)(x)

net = Model([input_a, input_b], out)
net.summary()


anchor_in = Input(shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH))
pos_in = Input(shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH))
neg_in = Input(shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH))

anchor_pos_out = net([anchor_in, pos_in])
pos_anchor_out = net([pos_in, anchor_in])
anchor_neg_out = net([anchor_in, neg_in])
neg_anchor_out = net([neg_in, anchor_in])

net_out = concatenate([anchor_pos_out, pos_anchor_out, anchor_neg_out, neg_anchor_out], axis=0, name='net_out')

model = Model(inputs=[anchor_in, pos_in, neg_in], outputs=net_out)
model.compile(optimizer=Adam(lr=0.0001), loss=hinge_loss)
model.summary()

def get_similarity_from_independent_model(textA, textB):
    result = model.predict_on_batch([np.reshape(get_ready_vector(textA), (1,MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH)),
                                   np.reshape(get_ready_vector(textB), (1,MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH)),
                                   np.reshape(get_ready_vector(textB), (1,MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH))])
    return result[0][0]


def epoch_test(epoch, logs):
    print(get_similarity_from_independent_model('Abdeckkappen Ø 5 mm weiß, 20 Stück', 'Abdeckkappe 5 mm weiss'))
    print(get_similarity_from_independent_model('Abdeckkappen Ø 5 mm weiß, 20 Stück', 'Abdeckkappe 6 mm weiss'))
    print('\n\n')

    test = 'Teppich MICHALSKY München anthrazit 133x190 cm'
    print(get_similarity_from_independent_model(test, 'Teppich MM München 133x190cm anthrazit'))
    print(get_similarity_from_independent_model('Teppich MM München 133x190cm anthrazit', test))
    print('\n')
    print(get_similarity_from_independent_model(test, 'Vliesfotot. 184x248 Brooklyn'))
    print(get_similarity_from_independent_model('Vliesfotot. 184x248 Brooklyn', test))
    print('\n')
    print(get_similarity_from_independent_model('Vliesfotot. 184x248 Brooklyn', 'Vliesfotot. 184x248 Brooklyn'))


epoch_end_callback = keras.callbacks.LambdaCallback(on_epoch_end=epoch_test)

if TRAIN:
    # model = load_model('trained_models/model_arc2_02_concat.h5', custom_objects={'hinge_loss': hinge_loss})
    data_generator = Native_DataGenerator_for_IndependentModel(batch_size=BATCH_SIZE)
    model.fit_generator(generator=data_generator, shuffle=True, epochs=100, workers=16, use_multiprocessing=True, callbacks=[epoch_end_callback])
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
