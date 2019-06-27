from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input, concatenate
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, LambdaCallback
from data_utilities.generator import Native_DataGenerator_for_StructuralSimilarityModel
from encoder import encode_number
import numpy as np
DIM = 10




def create_network():

    input_a = Input(shape=(1, DIM, DIM))
    input_b = Input(shape=(1, DIM, DIM))

    def common_network():
        layers = [Conv2D(filters=100, kernel_size=(2, 2), kernel_initializer='glorot_uniform',
                         input_shape=(1, DIM, DIM), data_format='channels_first',
                         use_bias=True, activation='tanh', padding='same'),
                  MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_first'),
                  Conv2D(filters=100, kernel_size=(2, 2), kernel_initializer='glorot_uniform',
                         data_format='channels_first', use_bias=True,
                         activation='tanh', padding='same'),
                  #MaxPooling2D(pool_size=(1, 1), strides=None, padding='valid', data_format='channels_first')
                  ]

        def shared_layers(x):
            for layer in layers:
                x = layer(x)
            return x

        return shared_layers


    common_net = common_network()
    out_a = Flatten()(common_net(input_a))
    out_b = Flatten()(common_net(input_b))

    concat_out = concatenate([out_a, out_b])

    x = concat_out
    x = Dense(activation='relu', units=1000, use_bias=True)(x)
    x = Dense(activation='relu', units=500, use_bias=True)(x)
    x = Dense(activation='relu', units=100, use_bias=True)(x)
    x = Dense(activation='relu', units=10, use_bias=True)(x)
    out = Dense(activation='relu', units=1, use_bias=True)(x)

    model = Model(inputs=[input_a, input_b], outputs=out)

    return model



model = create_network()
model.summary()
model.compile(optimizer=Adam(lr=0.0001), loss='mean_squared_error')
tensorboard = TensorBoard(log_dir='./logs/model_structuralsimilarity', histogram_freq=0, batch_size=120,
                          write_graph=True, write_grads=False, write_images=False, embeddings_freq=0,
                          embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None,
                          update_freq='epoch')


def epoch_test(epoch, logs):
    print(model.predict([np.expand_dims(np.expand_dims(encode_number(1000), axis=0), axis=0), np.expand_dims(np.expand_dims(encode_number(1), axis=0), axis=0)]))
    print(model.predict([np.expand_dims(np.expand_dims(encode_number(100), axis=0), axis=0), np.expand_dims(np.expand_dims(encode_number(1), axis=0), axis=0)]))
    print(model.predict([np.expand_dims(np.expand_dims(encode_number(100), axis=0), axis=0), np.expand_dims(np.expand_dims(encode_number(10), axis=0), axis=0)]))
    print(model.predict([np.expand_dims(np.expand_dims(encode_number(100), axis=0), axis=0), np.expand_dims(np.expand_dims(encode_number(100), axis=0), axis=0)]))


epoch_end_callback = LambdaCallback(on_epoch_end=epoch_test)
data_generator = Native_DataGenerator_for_StructuralSimilarityModel(batch_size=120)
model.fit_generator(generator=data_generator, shuffle=True, epochs=100, workers=1, use_multiprocessing=False, callbacks=[tensorboard, epoch_end_callback])
model.save(filepath='trained_models/model_structuralsimilarity.h5')

