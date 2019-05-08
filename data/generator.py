from keras.utils import Sequence
import numpy as np
from data.datareader import read_dataset_data
from texttovector import get_ready_vector
from config.configurations import ELMO_VECTOR_LENGTH, MAX_TEXT_WORD_LENGTH



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


class DataGenerator_for_Arc2(Sequence):

    def __init__(self, batch_size):
        data = read_dataset_data()
        anchor, pos, neg = data[data.columns[0]].to_numpy(), data[data.columns[1]].to_numpy(), data[data.columns[2]].to_numpy()
        x_set = np.column_stack((anchor, pos, neg))
        y_set = np.zeros((x_set.shape[0]), dtype=float)
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        input, y = np.array([[get_combinations(get_ready_vector(sample[0]), get_ready_vector(sample[1]), max_text_length=MAX_TEXT_WORD_LENGTH, word_embedding_length=ELMO_VECTOR_LENGTH),
                 get_combinations(get_ready_vector(sample[0]), get_ready_vector(sample[2]), max_text_length=MAX_TEXT_WORD_LENGTH, word_embedding_length=ELMO_VECTOR_LENGTH)] for sample in batch_x]), batch_y
        return input, y