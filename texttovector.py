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


def get_ready_tensors(sentence):
    sentence = tu.pre_process_single_return(str(sentence))
    sentence_word_length = len(sentence.split())

    tensors = []
    masks = []
    tensors_r = []
    masks_r = []
    words = sentence.split()
    for i in range(0, MAX_TEXT_WORD_LENGTH):
        if i < len(words):
            m, m_r = encode_word(words[i])
        else:
            m, m_r = encode_word(' ') #padding
        t, t_mask = convert_to_tensor(m)
        t_r, t_r_mask = convert_to_tensor(m_r)
        tensors.append(t)
        tensors_r.append(t_r)
        masks.append(t_mask)
        masks_r.append(t_r_mask)

    return np.array(tensors), np.array(tensors_r), np.array(masks), np.array(masks_r)

'''
t, t_r, m, m_r = get_ready_tensors('all is well')
print(t.shape)
print(t_r.shape)

t_r = np.repeat(t_r, [30], axis=0)
print(t_r.shape)
t = np.repeat(np.expand_dims(t, axis=0), [30], axis=0)
print(t.shape)
t = np.reshape(t, (90, 360, 3, 3))
print(t.shape)

'''


