from elmoformanylangs import Embedder

embedder = Embedder('C:\\Users\\seyit\\PycharmProjects\\thesis\\elmoformanylangs\\elmo_german_embeddings')


def get_elmo_vector(text):
    return embedder.sents2elmo([[text]])