from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input, concatenate, Bidirectional, LSTM, Reshape, Layer
from keras.models import Model, load_model
from keras.optimizers import Adam, Adagrad
from keras.callbacks import TensorBoard, LambdaCallback
import keras.backend as K
import tensorflow as tf
from data_utilities.generator import Native_DataGenerator_for_StructuralSimilarityModel_LSTMEncoder3x3
from encoder import encode_number
import numpy as np
from encoder import encode_word, convert_to_tensor
import time
DIM = 18
EPSILON = 0.0000001


class SimilarityLayer(Layer):
    def __init__(self, num_outputs):
      super(SimilarityLayer, self).__init__()
      self.model_name = 'model_structuralsimilarity_similarityspace3x320190730170704.h5'
      self.num_outputs = num_outputs
      self.similarity = load_model('trained_models/' + self.model_name)

    def build(self, input_shape):
      super(SimilarityLayer, self).build(input_shape)

    def call(self, input):
      in1 = K.eval(input[0])
      in2 = K.eval(input[1])
      mask = K.eval(input[2])
      output = []
      sum = []
      count = 0
      for i in range(1):
          w1 = np.squeeze(in1[i, :, :])
          w2 = np.squeeze(in2[i, :, :])
          #print(mask.shape)
          #print(mask)
          non_zero = np.sum(mask, 1)#non_zero_mask(w1, w2)
          #print(non_zero)
          result = self.similarity.predict_on_batch([K.variable(w1), K.variable(w2)])
          #print(result)
          result = np.reshape(result, (100, 360))
          #print(result)
          result = np.sum(result, 1)/(non_zero + EPSILON)
          output.append(result)
      return K.variable(np.reshape(np.array(output), (10, 10)))

class EncodingLayer(Layer):
    def __init__(self, num_outputs):
        super(EncodingLayer, self).__init__()
        self.num_outputs = num_outputs
        self.encoder = load_model('trained_models\model_structuralsimilarity_autoencoder3x3_4dim_embeddings_encoder.h5')

    def build(self, input_shape):
        super(EncodingLayer, self).build(input_shape)

    def call(self, input):
        return self.encoder.predict_on_batch(K.eval(input))


encoder = load_model('trained_models\model_structuralsimilarity_autoencoder3x3_4dim_embeddings_encoder.h5')
model_name = 'model_structuralsimilarity_similarityspace3x320190730170704.h5'

similarity_model = load_model('trained_models/' + model_name)
for layer in similarity_model.layers:
    print(layer.get_config())
'''

def create_network():

    input_a = Input(shape=((36000, 9)))
    input_b = Input(shape=((36000, 9)))
    input_mask = Input(shape=((100, 360)))

    encoding_layer = EncodingLayer((36000, 4))
    encoding_layer.trainable = False
    #similarity_layer = SimilarityLayer((10, 10))
    #similarity_layer.trainable = False


    encoded_a = encoding_layer(input_a)
    encoded_b = encoding_layer(input_b)

    #out = similarity_layer([K.variable(encoded_a), K.variable(encoded_b), K.variable(input_mask)])
    out = concatenate([encoded_a, encoded_b])


    model = Model(inputs=[input_a, input_b, input_mask], outputs=out)

    return model



model = create_network()
model.summary()
model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error')

m = encode_word('baba', return_reverse=False)
s1, s1_mask = convert_to_tensor(m)
s1 = np.expand_dims(s1, 0)
s1 = np.repeat(s1, 100, 0)
#print(s1)
s1 = np.reshape(s1, (1, 36000, 9))
#print(s1)

m = encode_word('baba', return_reverse=False)
s2, s2_mask = convert_to_tensor(m)
s2 = np.expand_dims(s2, 0)
#print(s2)
s2 = np.repeat(s2, 100, 0)
s2 = np.reshape(s2, (1, 36000, 9))
#print(s2)
mask = np.logical_or(s1_mask, s2_mask)*1
mask = np.expand_dims(mask, 0)
#print(mask)
mask = np.repeat(mask, 100, 0)

print(s1.shape)
print(s1.dtype)

model.predict_on_batch([s1, s2, mask])
'''
