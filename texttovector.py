import numpy as np
import text_utilities as tu
#from elmo import __get_elmo_sentence_embedding, __get_elmo_sentence_embedding_on_batch
from glove import __get_glove_sentence_embedding
from fasttext import __get_fasttext_sentence_embedding
from encoder import encode_word, convert_to_tensor
from config.configurations import MAX_TEXT_WORD_LENGTH, EMBEDDER, BATCH_SIZE, EMBEDDING_LENGTH


def get_ready_vector(text, padding = True, embedder = EMBEDDER):
    text = tu.pre_process_single_return(str(text))
    text_word_length = len(text.split())


    if text_word_length == 0:
        text = 'noise'
        text_word_length = 1

    if embedder == 'ELMO':
        embedding = None#__get_elmo_sentence_embedding(text)
    elif embedder ==  'FASTTEXT':
        embedding = __get_fasttext_sentence_embedding(text)
    else:
        embedding = __get_glove_sentence_embedding(text)

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

def get_ready_vector_on_batch(texts, padding = True, embedder = EMBEDDER, batch_size = BATCH_SIZE):

    if embedder == 'ELMO':
        texts = [[word for word in tu.pre_process_single_return(text).split()] for text in texts]
        text_word_lengths = [len(text) for text in texts]
        texts_embbedings = None#__get_elmo_sentence_embedding_on_batch(texts)
    else:
        texts = [tu.pre_process_single_return(str(text)) for text in texts]
        text_word_lengths = [len(text.split()) for text in texts]
        texts_embbedings = __get_fasttext_sentence_embedding(texts) #TODO

    if padding:
        padded_embeddings = []
        padded = np.zeros((MAX_TEXT_WORD_LENGTH, EMBEDDING_LENGTH), dtype=float)
        for embedding, text_word_length in zip(texts_embbedings, text_word_lengths):
            if text_word_length == 1 or text_word_length == 0:
                embedding = np.reshape(embedding, (1, EMBEDDING_LENGTH))
                if text_word_length == 0:
                    text_word_length = 1
            padded[:text_word_length] = embedding[:MAX_TEXT_WORD_LENGTH]
            padded_embeddings.append(padded)
        return np.array(padded_embeddings)

    else:
        return texts_embbedings


def get_ready_tensor(word, return_reverse=True):
    m, m_r = encode_word(word, return_reverse=return_reverse)
    t = convert_to_tensor(m)
    t_r = convert_to_tensor(m_r)



