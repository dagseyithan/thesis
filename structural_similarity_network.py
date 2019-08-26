from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input, concatenate, Bidirectional, LSTM, Reshape, Layer, \
    ReLU, Lambda, add
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, LambdaCallback
import keras.backend as K
from config import configurations
import numpy as np
from encoder import encode_word, convert_to_tensor
import time

DIM = 9
EPSILON = configurations.EPSILON
BATCH_SIZE = configurations.BATCH_SIZE
ALPHABET_LENGTH = configurations.ALPHABET_LENGTH
MAX_TEXT_WORD_LENGTH = configurations.MAX_TEXT_WORD_LENGTH
MAX_WORD_CHARACTER_LENGTH = configurations.MAX_WORD_CHARACTER_LENGTH
WORD_TO_WORD_COMBINATIONS = MAX_TEXT_WORD_LENGTH * MAX_TEXT_WORD_LENGTH
WORD_TENSOR_DEPTH = int((ALPHABET_LENGTH * MAX_WORD_CHARACTER_LENGTH) / DIM)


class EncodingLayer(Layer):
    def __init__(self, num_outputs):
        super(EncodingLayer, self).__init__()
        self.num_outputs = num_outputs
        self.model = load_model(r'C:\Users\seyit\PycharmProjects\thesis\trained_models\model_structuralsimilarity_autoencoder3x3_4dim_embeddings_encoder.h5')

    def build(self, input_shape):
        super(EncodingLayer, self).build(input_shape)
        layer_weights = self.model.get_layer('dense_1').get_weights()
        self.dense_1 = Dense(90, weights=[layer_weights[0], layer_weights[1]])
        layer_weights = self.model.get_layer('dense_2').get_weights()
        self.dense_2 = Dense(30, weights=[layer_weights[0], layer_weights[1]])
        layer_weights = self.model.get_layer('dense_3').get_weights()
        self.dense_3 = Dense(4, weights=[layer_weights[0], layer_weights[1]])
        self.relu = ReLU(max_value=1.0)

    def call(self, input):
        x = self.dense_1(input)
        x = self.relu(x)
        x = self.dense_2(x)
        x = self.relu(x)
        x = self.dense_3(x)
        x = self.relu(x)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.num_outputs[1])


class ConvolutionalLayer(Layer):
    def __init__(self, num_outputs):
        super(ConvolutionalLayer, self).__init__()
        self.num_outputs = num_outputs
        self.num_filters = 50
        self.kernel_size = (2, 2)
        self.model = load_model(r'C:\Users\seyit\PycharmProjects\thesis\trained_models\model_structuralsimilarity_similarityspace3x320190730170704.h5')

    def build(self, input_shape):
        super(ConvolutionalLayer, self).build(input_shape)
        layer_weights = self.model.get_layer('conv2d_1').get_weights()
        self.conv2d_1 = Conv2D(filters=self.num_filters, kernel_size=self.kernel_size, input_shape=(1, 2, 2),
                               data_format='channels_first', use_bias=True, activation='relu', padding='valid',
                               weights=[layer_weights[0], layer_weights[1]])
        self.reshape = Reshape((1, 2, 2))

    def call(self, input):
        out = []
        for i in range(BATCH_SIZE):
            x = input[i,:,:]
            x = self.reshape(x)
            x = self.conv2d_1(x)
            out.append(x)
        out = K.stack(out, axis=0)
        out = K.reshape(out, (BATCH_SIZE, int(WORD_TENSOR_DEPTH * WORD_TO_WORD_COMBINATIONS), self.num_filters))
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.num_filters)


class MLP(Layer):
    def __init__(self, num_outputs):
        super(MLP, self).__init__()
        self.num_outputs = num_outputs
        self.model = load_model(r'C:\Users\seyit\PycharmProjects\thesis\trained_models\model_structuralsimilarity_similarityspace3x320190730170704.h5')

    def build(self, input_shape):
        super(MLP, self).build(input_shape)
        layer_weights = self.model.get_layer('dense_1').get_weights()
        self.dense_1 = Dense(100, activation='relu', weights=[layer_weights[0], layer_weights[1]])
        layer_weights = self.model.get_layer('dense_2').get_weights()
        self.dense_2 = Dense(50, activation='relu', weights=[layer_weights[0], layer_weights[1]])
        layer_weights = self.model.get_layer('dense_3').get_weights()
        self.dense_3 = Dense(25, activation='relu', weights=[layer_weights[0], layer_weights[1]])
        layer_weights = self.model.get_layer('dense_4').get_weights()
        self.dense_4 = Dense(5, activation='relu', weights=[layer_weights[0], layer_weights[1]])
        layer_weights = self.model.get_layer('dense_5').get_weights()
        self.dense_5  = Dense(1, activation='relu', weights=[layer_weights[0], layer_weights[1]])

    def call(self, input):
        out = []
        for i in range(BATCH_SIZE):
            x = input[i,:,:]
            x = self.dense_1(x)
            x = self.dense_2(x)
            x = self.dense_3(x)
            x = self.dense_4(x)
            x = self.dense_5(x)
            out.append(x)
        out = K.stack(out, axis=0)
        out = K.reshape(out, (BATCH_SIZE, MAX_TEXT_WORD_LENGTH * MAX_TEXT_WORD_LENGTH, WORD_TENSOR_DEPTH))
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], 1)


class NonZeroMaskMatrixMean(Layer):
    def __init__(self, num_outputs):
        super(NonZeroMaskMatrixMean, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        super(NonZeroMaskMatrixMean, self).build(input_shape)

    def call(self, input):
        matrices = input[0]
        mask = input[1]
        out = []
        for i in range(BATCH_SIZE):
            out.append(K.sum(matrices[i,:,:], axis=-1) / (K.sum(mask[i,:,:], axis=-1) + EPSILON))
        out = K.stack(out, axis=0)
        out = K.reshape(out, (BATCH_SIZE, MAX_TEXT_WORD_LENGTH, MAX_TEXT_WORD_LENGTH))
        return out

    def compute_output_shape(self, input_shape):
        return (BATCH_SIZE, MAX_TEXT_WORD_LENGTH, MAX_TEXT_WORD_LENGTH)


class MatrixMean(Layer):
    def __init__(self, num_outputs):
        super(MatrixMean, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        super(MatrixMean, self).build(input_shape)

    def call(self, input):
        matrices_AB = input[0]
        matrices_AB_r = input[1]
        out = []
        for i in range(BATCH_SIZE):
            out.append(add([matrices_AB[i,:,:], matrices_AB_r[i,:,:]]) * 0.5)
        out = K.stack(out, axis=0)
        out = K.reshape(out, (BATCH_SIZE, MAX_TEXT_WORD_LENGTH, MAX_TEXT_WORD_LENGTH))
        return out

    def compute_output_shape(self, input_shape):
        return (BATCH_SIZE, MAX_TEXT_WORD_LENGTH, MAX_TEXT_WORD_LENGTH)


def StructuralSimilarityNetwork():
    input_A = Input(shape=((36000, 9)))
    input_A_r = Input(shape=((36000, 9)))
    input_B = Input(shape=((36000, 9)))
    input_B_r = Input(shape=((36000, 9)))
    input_mask = Input(shape=((100, 360)))
    input_mask_r = Input(shape=((100, 360)))

    encoding_layer = EncodingLayer(num_outputs = (36000, 4))
    convolutional_layer = ConvolutionalLayer(num_outputs = (36000, 50))
    nonzero_mask_matrix_mean = NonZeroMaskMatrixMean((BATCH_SIZE, MAX_TEXT_WORD_LENGTH, MAX_TEXT_WORD_LENGTH))
    matrix_mean = MatrixMean((BATCH_SIZE, MAX_TEXT_WORD_LENGTH, MAX_TEXT_WORD_LENGTH))
    mlp = MLP(num_outputs = (36000, 1))
    encoding_layer.trainable = False
    convolutional_layer.trainable = False
    mlp.trainable = False
    nonzero_mask_matrix_mean.trainable = False

    encoded_A = encoding_layer(input_A)
    encoded_A_r = encoding_layer(input_A_r)
    encoded_B = encoding_layer(input_B)
    encoded_B_r = encoding_layer(input_B_r)
    conv_out_A = convolutional_layer(encoded_A)
    conv_out_A_r = convolutional_layer(encoded_A_r)
    conv_out_B = convolutional_layer(encoded_B)
    conv_out_B_r = convolutional_layer(encoded_B_r)

    out_AB = concatenate([conv_out_A, conv_out_B], axis=-1)
    out_AB = mlp(out_AB)
    out_AB = nonzero_mask_matrix_mean([out_AB, input_mask])

    out_AB_r = concatenate([conv_out_A_r, conv_out_B_r], axis=-1)
    out_AB_r = mlp(out_AB_r)
    out_AB_r = nonzero_mask_matrix_mean([out_AB_r, input_mask_r])

    out = matrix_mean([out_AB, out_AB_r])


    model = Model(inputs=[input_A, input_A_r, input_B, input_B_r, input_mask, input_mask_r], outputs=out, name='str_sim_net')
    model.summary()

    return model


'''
model = StructuralSimilarityNetwork()
model.summary()
model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error')

m = encode_word('baba', return_reverse=False)
s1, s1_mask = convert_to_tensor(m)
s1 = np.expand_dims(s1, 0)
s1 = np.repeat(s1, 100, 0)
#print(s1)
s1 = np.reshape(s1, (1, 36000, 9))
s1 = np.repeat(s1, BATCH_SIZE, 0)
#print(s1)

m = encode_word('zulababa', return_reverse=False)
s2, s2_mask = convert_to_tensor(m)
s2 = np.expand_dims(s2, 0)
#print(s2)
s2 = np.repeat(s2, 100, 0)
s2 = np.reshape(s2, (1, 36000, 9))
s2 = np.repeat(s2, BATCH_SIZE, 0)
#print(s2)
mask = np.logical_or(s1_mask, s2_mask)*1
mask = np.expand_dims(mask, 0)
#print(mask)
mask = np.repeat(mask, 100, 0)
mask = np.expand_dims(mask, 0)
mask = np.repeat(mask, BATCH_SIZE, 0)

#print(s1.shape)
#print(s2.shape)
print(mask.shape)

print('getting results')
#print(model.predict_on_batch([s1, s2, mask]))
print(model.predict_on_batch([s1, s2, mask]))
'''