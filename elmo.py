'''
from elmoformanylangs import Embedder
import platform
from config import configurations
import numpy as np

if configurations.EMBEDDER == 'ELMO':
    if platform.system() == 'Linux':
        if configurations.LANGUAGE == 'GERMAN':
            embedder = Embedder(configurations.LINUX_ELMO_GERMAN_EMBEDDINGS_MODEL_PATH, batch_size=configurations.BATCH_SIZE)
        elif configurations.LANGUAGE == 'ENGLISH':
            embedder = Embedder(configurations.LINUX_ELMO_ENGLISH_EMBEDDINGS_MODEL_PATH, batch_size=configurations.BATCH_SIZE)
    else:
        if configurations.LANGUAGE == 'GERMAN':
            embedder = Embedder(configurations.WINDOWS_ELMO_GERMAN_EMBEDDINGS_MODEL_PATH, batch_size=configurations.BATCH_SIZE)
        elif configurations.LANGUAGE == 'ENGLISH':
            embedder = Embedder(configurations.WINDOWS_ELMO_ENGLISH_EMBEDDINGS_MODEL_PATH, batch_size=configurations.BATCH_SIZE)
else:
    print('ERROR: ELMO is not set as the embedder!')


    print('ELMo ' + configurations.LANGUAGE + ' model has been loaded...')

def __get_elmo_word_embedding(word):

    return np.squeeze(np.array(embedder.sents2elmo([[word]], output_layer=-1)))


def __get_elmo_sentence_embedding(text):
    #param text: collection of words, as plain string
    #return: an array containing elmo embedding of each word in the given collection, respecting the order
    words = [[word] for word in text.split()]
    return np.squeeze(np.array(embedder.sents2elmo(words, output_layer=-1)))

def __get_elmo_sentence_embedding_on_batch(texts):

    #:param text: collection of texts as list of string lists
    #:return: an array containing elmo embedding of each word in the given collection, respecting the order

    return np.squeeze(np.array(embedder.sents2elmo(texts, output_layer=-1)))
'''