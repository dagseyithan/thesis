from keras.utils.data_utils import Sequence
import numpy as np
from data_utilities.datareader import read_dataset_data
from texttovector import get_ready_vector
from config.configurations import ELMO_VECTOR_LENGTH, MAX_TEXT_WORD_LENGTH, EMBEDDER, FASTTEXT_VECTOR_LENGTH

if EMBEDDER == 'FASTTEXT':
    EMBEDDING_LENGTH = FASTTEXT_VECTOR_LENGTH
else:
    EMBEDDING_LENGTH = ELMO_VECTOR_LENGTH


def get_combinations(vec_A, vec_B, max_text_length, word_embedding_length, window_size = 3):
    combined = []
    i, j = 0, 0
    while i+window_size <= max_text_length:
        while j+window_size <= max_text_length:
            stacked = np.vstack((vec_A[i:i+window_size], vec_B[j:j+window_size]))
            combined.append(list(stacked))
            j += 1
        j = 0
        i += 1
    combined = np.array(combined)
    return np.reshape(combined, (combined.shape[0] * combined.shape[1], word_embedding_length))


def get_concat(vec_A, vec_B, max_text_length, word_embedding_length, window_size = 3):
    return np.concatenate((vec_A, vec_B))


class Native_DataGenerator_for_Arc2(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        #data = read_dataset_data('train')
        #anchor, pos, neg = data[data.columns[0]].to_numpy(), data[data.columns[1]].to_numpy(), data[data.columns[2]].to_numpy()
        #x_set = np.column_stack((anchor[0:201], pos[0:201], neg[0:201]))
        #y_set = np.zeros((x_set.shape[0]), dtype=float)
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size))) - 1

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        '''
        anchor_pos = np.array([get_combinations(get_ready_vector(sample[0]), get_ready_vector(sample[1]),
                                                max_text_length=MAX_TEXT_WORD_LENGTH,
                                                word_embedding_length=EMBEDDING_LENGTH) for sample in batch_x])
        anchor_neg = np.array([get_combinations(get_ready_vector(sample[0]), get_ready_vector(sample[2]),
                                                max_text_length=MAX_TEXT_WORD_LENGTH,
                                                word_embedding_length=EMBEDDING_LENGTH) for sample in batch_x])

        
        '''
        anchor_pos = np.array([get_concat(get_ready_vector(sample[0]), get_ready_vector(sample[1]),
                                                max_text_length=MAX_TEXT_WORD_LENGTH,
                                                word_embedding_length=EMBEDDING_LENGTH) for sample in batch_x])

        anchor_neg = np.array([get_concat(get_ready_vector(sample[0]), get_ready_vector(sample[2]),
                                                max_text_length=MAX_TEXT_WORD_LENGTH,
                                                word_embedding_length=EMBEDDING_LENGTH) for sample in batch_x])
                                                

        return [anchor_pos, anchor_neg], batch_y


def DataGenerator_for_Arc2(batch_size):
    data = read_dataset_data('train')
    anchor, pos, neg = data[data.columns[0]].to_numpy(), data[data.columns[1]].to_numpy(), data[data.columns[2]].to_numpy()
    x = np.column_stack((anchor, pos, neg))
    y = np.zeros((x.shape[0]), dtype=float)
    DATASET_SIZE = x.shape[0]

    while True:
        for i in range(0, DATASET_SIZE):
            batch_x = x[i * batch_size:(i + 1) * batch_size]
            batch_y = y[i * batch_size:(i + 1) * batch_size]

            anchor_pos = np.array([get_combinations(get_ready_vector(sample[0]), get_ready_vector(sample[1]),
                                                    max_text_length=MAX_TEXT_WORD_LENGTH,
                                                    word_embedding_length=EMBEDDING_LENGTH) for sample in batch_x])
            anchor_neg = np.array([get_combinations(get_ready_vector(sample[0]), get_ready_vector(sample[2]),
                                                    max_text_length=MAX_TEXT_WORD_LENGTH,
                                                    word_embedding_length=EMBEDDING_LENGTH) for sample in batch_x])

            yield [anchor_pos, anchor_neg], np.array(batch_y)