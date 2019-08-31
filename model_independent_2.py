from keras.layers import Conv1D, Conv2D, MaxPooling2D, Dense, Reshape, Flatten, Input, concatenate, \
    Lambda, Bidirectional, BatchNormalization, LSTM
from keras.models import Model, load_model
import keras.callbacks
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
from config.configurations import MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH, BATCH_SIZE
from data_utilities.generator import Native_DataGenerator_for_IndependentModel, Native_DataGenerator_for_SemanticSimilarityNetwork
from texttovector import get_ready_vector
from scipy.spatial.distance import cosine
from sklearn.preprocessing import minmax_scale
from data_utilities.datareader import read_sts_data
from texttovector import get_ready_vector
import numpy as np

def tf_pearson(y_true, y_pred):
    return tf.contrib.metrics.streaming_pearson_correlation(y_pred, y_true)[1]

TRAIN = True


def hinge_loss(y_true, y_pred, N=1000, beta=1000.0, epsilon=K.epsilon()):
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
                  input_shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH), use_bias=True, activation='tanh', padding='same')
conv3k = Conv1D(filters=400, kernel_size=3, kernel_initializer='glorot_uniform',
                  input_shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH), use_bias=True, activation='tanh', padding='same')
conv2k_out = conv2k(input_a)
conv3k_out = conv3k(input_a)
concat_convs_a = concatenate([conv2k_out, conv3k_out], axis=1)

lstm_concat = Bidirectional(LSTM(units=200, activation='tanh', return_sequences=True, input_shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH)), merge_mode='concat')
lstm_a = Bidirectional(LSTM(units=400, activation='tanh', return_sequences=True, input_shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH)), merge_mode='concat')
lstm_last = Bidirectional(LSTM(units=25, activation='tanh',return_sequences=True, input_shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH)), merge_mode='concat')
lstm_concat_out = lstm_concat(concat_convs_a)
lstm_a_out = lstm_a(input_a)


def common_network():
    layers = [Conv2D(filters=100, kernel_size=(2, 2), kernel_initializer='glorot_uniform', data_format='channels_first',
                     use_bias=True, activation='tanh', padding='same'),
              MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid',data_format='channels_first'),
              Conv2D(filters=100, kernel_size=(2, 2), activation='tanh', kernel_initializer='glorot_uniform',
                     data_format='channels_first', padding='same'),
              MaxPooling2D(pool_size=(5, 5), strides=None, padding='valid', data_format='channels_first')
    ]

    def shared_layers(x):
        for layer in layers:
            x = layer(x)
        return x

    return shared_layers

common_net = common_network()
lstm_concat_out = Reshape((2, MAX_TEXT_WORD_LENGTH, 400))(lstm_concat_out)
lstm_a_out = Reshape((2, MAX_TEXT_WORD_LENGTH, 400))(lstm_a_out)
common_concat_out = common_net(lstm_concat_out)
common_lstm_out = common_net(lstm_a_out)
common_common_out = concatenate([common_concat_out, common_lstm_out], axis=1)
K.int_shape(common_common_out)
common_common_out = Reshape((MAX_TEXT_WORD_LENGTH, 800))(common_common_out)
lstm_a_out = Reshape((MAX_TEXT_WORD_LENGTH, 800))(lstm_a_out)

last_concat =  concatenate([common_common_out, lstm_a_out], axis=1)
#lstm_last_out = lstm_last(last_concat)

x = Flatten()(last_concat)#(lstm_last_out)
out = Dense(units=1000, activation='tanh')(x)

net = Model([input_a], out)
net.summary()
#plot_model(net, 'model_independent2.png', show_shapes=True, show_layer_names=True)


embedded_sentence_A = Input(shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH))  # (10, 300)
embedded_sentence_B = Input(shape=(MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH))  # (10, 300)

A_out = net(embedded_sentence_A)
B_out = net(embedded_sentence_B)

concat = concatenate([A_out, B_out], axis=-1)
concat = Dense(units=2000, activation='tanh')(concat)
concat = Dense(units=1000, activation='tanh')(concat)
concat = Dense(units=500, activation='tanh')(concat)
concat = Dense(units=100, activation='tanh')(concat)
concat = Dense(units=50, activation='tanh')(concat)
out = Dense(units=1, activation='sigmoid')(concat)

#net_out = concatenate([embedded_sentence_A, embedded_sentence_B, neg_out], axis=-1, name='net_out')

model = Model(inputs=[embedded_sentence_A, embedded_sentence_B], outputs=out)
model.compile(optimizer=Adam(lr=0.01), loss='mean_squared_error', metrics=['mean_absolute_error', tf_pearson])
model.summary()

def get_similarity_from_independent_model_2(textA, textB):
    return model.predict_on_batch([np.reshape(get_ready_vector(textA), (1,MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH)),
                                   np.reshape(get_ready_vector(textB), (1,MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH))])


def epoch_test(epoch, logs):
    print(get_similarity_from_independent_model_2('A girl is styling her hair', 'A girl is brushing her hair.'))
    print(get_similarity_from_independent_model_2('A girl is styling her hair', 'A girl is styling her hair.'))
    print(get_similarity_from_independent_model_2('A baby panda goes down a slide', 'A panda slides down a slide.'))
    print(get_similarity_from_independent_model_2('A man is kicking pots of water.', 'A man is picking flowers.'))

    print('\n\n')
    '''

    test = 'Teppich MICHALSKY München anthrazit 133x190 cm'
    print(get_similarity_from_independent_model_2(test, 'Teppich MM München 133x190cm anthrazit'))
    print(get_similarity_from_independent_model_2('Teppich MM München 133x190cm anthrazit', test))
    print('\n')
    print(get_similarity_from_independent_model_2(test, 'Vliesfotot. 184x248 Brooklyn'))
    print(get_similarity_from_independent_model_2('Vliesfotot. 184x248 Brooklyn', test))
    print('\n')
    print(get_similarity_from_independent_model_2('Vliesfotot. 184x248 Brooklyn', 'Vliesfotot. 184x248 Brooklyn'))
    '''
test_size = 1300
dataset_size = 5700
sentences_A, sentences_B, scores = read_sts_data('test')
sentences_A = np.array([get_ready_vector(sentence) for sentence in sentences_A[0:test_size]])
sentences_B = np.array([get_ready_vector(sentence) for sentence in sentences_B[0:test_size]])
scores = minmax_scale(scores, feature_range=(0, 0.99))
scores = scores[0:test_size]

val_data = [[sentences_A, sentences_B], scores]




epoch_end_callback = keras.callbacks.LambdaCallback(on_epoch_end=epoch_test)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=5, factor=0.1, verbose=1, min_lr=0.0001)
#checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath='trained_models/model_independent_2_02_RegularRNN_Fasttext_mixedmargin_update00.h5', period=1)

if TRAIN:
    # model = load_model('trained_models/model_arc2_02_concat.h5', custom_objects={'hinge_loss': hinge_loss})
    data_generator = Native_DataGenerator_for_SemanticSimilarityNetwork(batch_size=BATCH_SIZE, dataset_size=dataset_size)
    K.get_session().run(tf.local_variables_initializer())
    model.fit_generator(generator=data_generator,validation_data=val_data, shuffle=True, epochs=50, workers=14, use_multiprocessing=True,
                        callbacks=[reduce_lr])
    model.save('trained_models/model_independent_2_01_Fasttext_mixedmargin.h5')
else:
    model = load_model('trained_models\model_independent_2_02_RegularRNN_Fasttext_mixedmargin.h5', custom_objects={'hinge_loss': hinge_loss})
    model.summary()


'''

print('\n')

print(get_similarity_from_independent_model('Unterkonstruktion tepro Pent Roof 6x4 193 x 112 cm, silber', 'Unterk.Pent Roof 6x4für 7116/7209/7236'))
print(get_similarity_from_independent_model('Unterkonstruktion tepro Pent Roof 6x4 193 x 112 cm, silber', 'Unterkonstruktion tepro Riverton 6x4 192,2 x 112,1 cm, silber'))
print(get_similarity_from_independent_model('Unterkonstruktion tepro Pent Roof 6x4 193 x 112 cm, silber', 'Feuchtraumkabel NYM-J 3G1,5 10M'))
print(get_similarity_from_independent_model('Unterkonstruktion tepro Pent Roof 6x4 193 x 112 cm, silber', 'Unterk.Riverton 6x4 HAUSTYP1/7124/7240'))
print(get_similarity_from_independent_model('Unterkonstruktion tepro Riverton 6x4 192,2 x 112,1 cm, silber', 'Unterk.Pent Roof 6x4für 7116/7209/7236'))
print(get_similarity_from_independent_model('Unterkonstruktion tepro Riverton 6x4 192,2 x 112,1 cm, silber', 'Unterk.Riverton 6x4 HAUSTYP1/7124/7240'))
print(get_similarity_from_independent_model('Unterk.Riverton 6x4 HAUSTYP1/7124/7240', 'Unterkonstruktion tepro Pent Roof 6x4 193 x 112 cm, silber'))
print(get_similarity_from_independent_model('Unterk.Riverton 6x4 HAUSTYP1/7124/7240', 'Unterkonstruktion tepro Riverton 6x4 192,2 x 112,1 cm, silber'))
'''