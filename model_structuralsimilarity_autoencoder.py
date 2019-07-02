from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input, concatenate, Bidirectional, LSTM, Reshape, UpSampling2D
from keras.models import Model, load_model
from keras.optimizers import Adam, Adagrad
from keras.callbacks import TensorBoard, LambdaCallback
import keras.backend as K
import tensorflow as tf
from data_utilities.generator import Native_DataGenerator_for_StructuralSimilarityModel_Autoencoder
from encoder import encode_number
import numpy as np
DIM = 30


def create_network():
    input = Input(shape=(DIM * DIM,))

    def common_network():
        layers = [Dense(3000, activation='relu'),
                  Dense(1000, activation='relu'),
                  Dense(900, activation='relu'),
                  Dense(600, activation='relu'),
                  Dense(300, activation='relu'),
                  Dense(100, activation='relu'),
                  Dense(30, activation='relu'),
                  Dense(10, activation='relu'),
                  Dense(3, activation='relu'),
                  Dense(10, activation='relu'),
                  Dense(30, activation='relu'),
                  Dense(100, activation='relu'),
                  Dense(300, activation='relu'),
                  Dense(600, activation='relu'),
                  Dense(900, activation='relu'),
                  Dense(1000, activation='relu'),
                  Dense(3000, activation='relu'),
                  Dense(900, activation='sigmoid'),
        ]

        def shared_layers(x):
            for layer in layers:
                x = layer(x)
            return x

        return shared_layers

    common_net = common_network()
    out = common_net(input)
    model = Model(inputs=[input], outputs=out)

    return model


model = create_network()
model.summary()
model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy')
model_name = 'model_structuralsimilarity_autoencoder'
tensorboard = TensorBoard(log_dir='./logs/model_structuralsimilarity/' + model_name, histogram_freq=0, batch_size=120,
                          write_graph=True, write_grads=False, write_images=False, embeddings_freq=0,
                          embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None,
                          update_freq='epoch')

data_generator = Native_DataGenerator_for_StructuralSimilarityModel_Autoencoder(batch_size=120)
model.fit_generator(generator=data_generator, shuffle=True, epochs=300, workers=4, use_multiprocessing=False, callbacks=[tensorboard])
model.save(filepath='trained_models/'+model_name+'.h5')

