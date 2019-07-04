from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, LambdaCallback, ReduceLROnPlateau
from keras.initializers import TruncatedNormal
from keras.layers.advanced_activations import ReLU
from data_utilities.generator import Native_DataGenerator_for_StructuralSimilarityModel_Autoencoder
import numpy as np
DIM = 9

relu = ReLU()
relu.__name__ = 'relu'

encoder_input = Input(shape=(DIM,))
x = Dense(1000)(encoder_input)
x = ReLU(max_value=1.0)(x)
x = Dense(600)(x)
x = ReLU(max_value=1.0)(x)
x = Dense(300)(x)
x = ReLU(max_value=1.0)(x)
x = Dense(100)(x)
x = ReLU(max_value=1.0)(x)
x = Dense(10)(x)
x = ReLU(max_value=1.0)(x)
x = Dense(3)(x)
encoder_output = ReLU(max_value=1.0)(x)

encoder = Model(encoder_input, encoder_output, name='encoder')
encoder.summary()

decoder_input = Input(shape=(3,))
x = Dense(10)(decoder_input)
x = ReLU(max_value=1.0)(x)
x = Dense(100)(x)
x = ReLU(max_value=1.0)(x)
x = Dense(300)(x)
x = ReLU(max_value=1.0)(x)
x = Dense(600)(x)
x = ReLU(max_value=1.0)(x)
x = Dense(1000)(x)
x = ReLU(max_value=1.0)(x)
x = Dense(9)(x)
decoder_output = ReLU(max_value=1.0)(x)


decoder = Model(decoder_input, decoder_output, name='decoder')
decoder.summary()

autoencoder_input = Input(shape=(DIM,))
encoded = encoder(autoencoder_input)
decoded = decoder(encoded)
autoencoder = Model(autoencoder_input, decoded, name='autoencoder')
autoencoder.summary()


autoencoder.compile(optimizer=Adam(lr=0.001, clipvalue=1.0), loss='mean_squared_error')
model_name = 'model_structuralsimilarity_autoencoder'
tensorboard = TensorBoard(log_dir='./logs/model_structuralsimilarity/' + model_name, histogram_freq=0, batch_size=300,
                          write_graph=True, write_grads=False, write_images=False, embeddings_freq=0,
                          embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None,
                          update_freq='epoch')

def epoch_test(epoch, logs):
    st = np.random.randint(2, size=(1, 9)).astype(np.float)
    print(st)
    print(autoencoder.predict(st))

    st = np.random.randint(2, size=(1, 9)).astype(np.float)
    print(st)
    print(autoencoder.predict(st))


epoch_end_callback = LambdaCallback(on_epoch_end=epoch_test)
reduce_lr = ReduceLROnPlateau(monitor='loss', patience=5, factor=0.1, verbose=1, min_lr=0.0001)

data_generator = Native_DataGenerator_for_StructuralSimilarityModel_Autoencoder(batch_size=200)
autoencoder.fit_generator(generator=data_generator, shuffle=True, epochs=500, workers=4, use_multiprocessing=False, callbacks=[tensorboard, epoch_end_callback, reduce_lr])
autoencoder.save(filepath='trained_models/'+model_name+'.h5')
encoder.save(filepath='trained_models/'+model_name+ '_encoder'+'.h5')
decoder.save(filepath='trained_models/'+model_name+'_decoder'+'.h5')