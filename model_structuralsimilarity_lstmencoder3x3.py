from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input, concatenate, Bidirectional, LSTM, Reshape
from keras.models import Model, load_model
from keras.optimizers import Adam, Adagrad
from keras.callbacks import TensorBoard, LambdaCallback
import keras.backend as K
import tensorflow as tf
from data_utilities.generator import Native_DataGenerator_for_StructuralSimilarityModel_LSTMEncoder3x3
from encoder import encode_number
import numpy as np
import time
DIM = 18




def create_network():

    input_a = Input(shape=(DIM, 1))


    def common_network():
        layers = [LSTM(units=18, activation='relu', return_sequences=True, input_shape=(DIM, 1)),
                  LSTM(units=18, activation='relu', return_sequences=True),
                  LSTM(units=18, activation='relu', return_sequences=True),
                  LSTM(units=18, activation='relu', return_sequences=True),
                  LSTM(units=18, activation='relu', return_sequences=True),
                  LSTM(units=18, activation='relu', return_sequences=True),
                  LSTM(units=18, activation='relu', return_sequences=True),
                  LSTM(units=18, activation='relu', return_sequences=True),
                  LSTM(units=1, activation='relu', return_sequences=False)
                  ]

        def shared_layers(x):
            for layer in layers:
                x = layer(x)
            return x

        return shared_layers


    common_net = common_network()
    out = common_net(input_a)


    model = Model(inputs=[input_a], outputs=out)

    return model



model = create_network()
model.summary()
model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error')
model_name = 'model_structuralsimilarity_lstmencoder3x3'
tensorboard = TensorBoard(log_dir='./logs/model_structuralsimilarity/' + model_name+time.strftime("%Y%m%d%H%M%S"), histogram_freq=0, batch_size=20,
                          write_graph=True, write_grads=False, write_images=False, embeddings_freq=0,
                          embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None,
                          update_freq='epoch')


def epoch_test(epoch, logs):
    print(model.predict([np.expand_dims(np.expand_dims(encode_number(1000), axis=0), axis=0), np.expand_dims(np.expand_dims(encode_number(1), axis=0), axis=0)]))
    print(model.predict([np.expand_dims(np.expand_dims(encode_number(1000), axis=0), axis=0), np.expand_dims(np.expand_dims(encode_number(1), axis=0), axis=0)]))
    print(model.predict([np.expand_dims(np.expand_dims(encode_number(1000), axis=0), axis=0), np.expand_dims(np.expand_dims(encode_number(10), axis=0), axis=0)]))
    print(model.predict([np.expand_dims(np.expand_dims(encode_number(1000), axis=0), axis=0), np.expand_dims(np.expand_dims(encode_number(20), axis=0), axis=0)]))
    print(model.predict([np.expand_dims(np.expand_dims(encode_number(1000), axis=0), axis=0), np.expand_dims(np.expand_dims(encode_number(30), axis=0), axis=0)]))
    print(model.predict([np.expand_dims(np.expand_dims(encode_number(1000), axis=0), axis=0), np.expand_dims(np.expand_dims(encode_number(1000), axis=0), axis=0)]))


epoch_end_callback = LambdaCallback(on_epoch_end=epoch_test)
data_generator = Native_DataGenerator_for_StructuralSimilarityModel_LSTMEncoder3x3(batch_size=120)
model.fit_generator(generator=data_generator, shuffle=True, epochs=300, workers=4, use_multiprocessing=False, callbacks=[tensorboard])
model.save(filepath='trained_models/'+model_name+'.h5')

