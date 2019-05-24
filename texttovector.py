import numpy as np
import text_utilities as tu
from elmo import __get_elmo_sentence_embedding
from fasttext import __get_fasttext_sentence_embedding
from config.configurations import ELMO_VECTOR_LENGTH, MAX_TEXT_WORD_LENGTH, FASTTEXT_VECTOR_LENGTH, EMBEDDER
import pprint as pp



def get_ready_vector(text, padding = True, embedder = EMBEDDER):
    text = tu.pre_process_single_return(text)
    text_word_length = len(text.split())

    if text_word_length == 0:
        text = 'noise'
        text_word_length = 1

    if embedder == 'ELMO':
        EMBEDDING_LENGTH = ELMO_VECTOR_LENGTH
        embedding = __get_elmo_sentence_embedding(text)
    else:
        EMBEDDING_LENGTH = FASTTEXT_VECTOR_LENGTH
        embedding = __get_fasttext_sentence_embedding(text)

    if padding:
        padded = np.zeros((MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH), dtype=float)
        if text_word_length == 1 or text_word_length == 0:
            embedding = np.reshape(embedding, (1, EMBEDDING_LENGTH))
            if text_word_length == 0:
                text_word_length = 1
        padded[:text_word_length] = embedding[:MAX_TEXT_WORD_LENGTH]
        return padded
    else:
        return embedding