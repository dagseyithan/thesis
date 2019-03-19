import fastText
import platform
import configurations
import numpy as np

if platform.system() == 'Linux':
    if configurations.LANGUAGE == 'GERMAN':
        model = fastText.load_model(configurations.LINUX_FASTTEXT_GERMAN_EMBEDDINGS_MODEL_PATH)
    elif configurations.LANGUAGE == 'ENGLISH':
        model = fastText.load_model(configurations.LINUX_FASTTEXT_ENGLISH_EMBEDDINGS_MODEL_PATH)
else:
    if configurations.LANGUAGE == 'GERMAN':
        model = fastText.load_model(configurations.WINDOWS_FASTTEXT_GERMAN_EMBEDDINGS_MODEL_PATH)
    elif configurations.LANGUAGE == 'ENGLISH':
        model = fastText.load_model(configurations.WINDOWS_FASTTEXT_ENGLISH_EMBEDDINGS_MODEL_PATH)

print('fastText model has been loaded...')


def __get_fasttext_word_embedding(text):
    return model.get_word_vector(text)


def __get_fasttext_sentence_embedding(sentence):
    return __get_fasttext_embeddings_average(__get_fasttext_word_embeddings(sentence))


def __get_fasttext_word_embeddings(sentence):
    return np.array([__get_fasttext_word_embedding(word) for word in sentence.split()])


def __get_fasttext_embeddings_average(sentence_vectors):
    num_words = np.float32(len(sentence_vectors))
    sentence_vectors = np.array(sentence_vectors)
    return np.divide(sentence_vectors.sum(axis=0), num_words)