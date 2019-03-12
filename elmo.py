from elmoformanylangs import Embedder
import platform

if platform.system() == 'Linux':
    embedder = Embedder('/home/sdag/PycharmProjects/thesis/elmoformanylangs/elmo_german_embeddings')
else:
    embedder = Embedder('C:\\Users\\seyit\\PycharmProjects\\thesis\\elmoformanylangs\\elmo_german_embeddings')


print('ELMo model has been loaded...')

def get_elmo_word_embedding(text):
    '''
    output_layer: the target layer to output.

    0 for the word encoder
    1 for the first LSTM hidden layer
    2 for the second LSTM hidden layer
    -1 for an average of 3 layers. (default)
    -2 for all 3 layers
    '''
    return embedder.sents2elmo([[text]], output_layer=0)