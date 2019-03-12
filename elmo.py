from elmoformanylangs import Embedder
import platform

if platform.system() == 'Linux':
    embedder = Embedder('/home/sdag/PycharmProjects/thesis/elmoformanylangs/elmo_german_embeddings')
else:
    embedder = Embedder('C:\\Users\\seyit\\PycharmProjects\\thesis\\elmoformanylangs\\elmo_german_embeddings')


print('ELMo model has been loaded...')

def get_elmo_embedding(text):
    return embedder.sents2elmo([[text]])