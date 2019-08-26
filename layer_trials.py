from keras.models import load_model
import numpy as np
import keras
import keras.backend as K
from encoder import encode_word, convert_to_tensor
from config.configurations import MAX_TEXT_WORD_LENGTH

batch_size = 1
EPSILON = 0.0000001


def non_zero_mask(a, b):
    zero_vector = np.array([0.15910167, 0.14220619, 0., 0.01655571], dtype=np.float32)
    count = 0
    for i in range(36000):
        if not (np.allclose(a[i], zero_vector)) or not (np.allclose(b[i], zero_vector)):
            count+=1
    return count + EPSILON


class SimilarityLayer(keras.layers.Layer):
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

class EncodingLayer(keras.layers.Layer):
  def __init__(self, num_outputs):
    super(EncodingLayer, self).__init__()
    self.num_outputs = num_outputs
    self.encoder = load_model('trained_models\model_structuralsimilarity_autoencoder3x3_4dim_embeddings_encoder.h5')

  def call(self, input):
    input = K.eval(input)
    output = []
    for i in range(1):
      slice = np.squeeze(input[i, :, :])
      output.append(self.encoder.predict_on_batch(K.variable(slice)))
    return K.variable(np.array(output))


encoding_layer = EncodingLayer((1, 360, 4))
similarity_layer = SimilarityLayer((10, 10))
'''
a = np.array(np.ones((1,9)))
b = np.array(np.zeros((batch_size,9)))
a = K.repeat_elements(K.variable(a), 2, 0)
print(a)
b = K.variable(b)
a_embedded = K.eval(embedding_layer(a))
b_embedded = K.eval(embedding_layer(b))
print(a_embedded)
print(b_embedded)
'''

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

print(mask.shape)


s1 = K.variable(s1)
s2 = K.variable(s2)
mask = K.variable(mask)

embedded_s1 = encoding_layer(s1)
embedded_s2 = encoding_layer(s2)

print('getting similarity...')
out = similarity_layer([embedded_s1, embedded_s2, mask])
print(K.eval(out))



'''
a = np.array([np.ones((9,))])
a = encoder3x3.predict(a)

b = np.array([np.ones((9,))])
b = encoder3x3.predict(b)

print(model.predict([a, b])[0][0])
'''

#print(model.predict_on_batch([a_embedded, b_embedded]))