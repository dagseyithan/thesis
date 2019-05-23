import numpy as np
import text_utilities as tu
from elmo import __get_elmo_sentence_embedding
from fasttext import __get_fasttext_sentence_embedding
from config.configurations import ELMO_VECTOR_LENGTH, MAX_TEXT_WORD_LENGTH, FASTTEXT_VECTOR_LENGTH, EMBEDDER



def get_ready_vector(text, padding = True, embedder = EMBEDDER):
    text = tu.pre_process_single_return(text)
    text_word_length = len(text.split())


    if embedder == 'ELMO':
        EMBEDDING_LENGTH = ELMO_VECTOR_LENGTH
        embedding = __get_elmo_sentence_embedding(text)
    else:
        EMBEDDING_LENGTH = FASTTEXT_VECTOR_LENGTH
        embedding = __get_fasttext_sentence_embedding(text)
    if padding:
        padded = np.zeros((MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH), dtype=float)
        if text_word_length == 1:
            embedding = np.reshape(embedding, (1, EMBEDDING_LENGTH))
        padded[:text_word_length] = embedding[:MAX_TEXT_WORD_LENGTH]
        return padded
    else:
        return embedding