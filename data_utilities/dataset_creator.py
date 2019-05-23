import random
from data_utilities.datareader import read_original_products_data, read_dataset_data
import h5py
from config import configurations
import numpy as np
from data_utilities.generator import get_combinations
from texttovector import get_ready_vector



def split_dataset(run = False):
    if run:
        data = read_dataset_data()
        anchor, pos, neg = data[data.columns[0]], data[data.columns[1]], data[data.columns[2]]
        rand = random.sample(range(0, len(anchor)), 1000)

        with open('dataset_train.csv','w', encoding='utf-8') as file_train, open('dataset_test.csv','w', encoding='utf-8') as file_test:
            for i in range(0, len(anchor)):
                if i in rand:
                    file_test.write(anchor[i]+';'+pos[i]+';'+neg[i]+'\n')
                else:
                    file_train.write(anchor[i] + ';' + pos[i] + ';' + neg[i] + '\n')
        file_test.close()
        file_train.close()
    return None


def create_dataset(run = False):
    if run:
        Data = read_original_products_data()
        ProductX, ProductY = Data[Data.columns[0]], Data[Data.columns[4]]

        with open('dataset.csv','w', encoding='utf-8') as file:
            rand = 0
            for i in range(0, len(ProductX)):
                rand = random.randint(0, len(ProductX) - 1)
                while rand == i:
                    rand = random.randint(0, len(ProductX) - 1)
                file.write(ProductX[i]+';'+ProductY[i]+';'+ProductY[rand]+'\n')
        file.close()
    return None


'''
def create_preprocessed_dataset():
    F = h5py.File('C:\\Users\\seyit\\PycharmProjects\\thesis\\data\\preprocessed_dataset.h5', "w")
    train_data = F.create_group("train")
    #validation_data = F.create_group("validation")
    train_pos = train_data.create_dataset("positive", (0, 1944, configurations.ELMO_VECTOR_LENGTH), maxshape=(None, 1944, configurations.ELMO_VECTOR_LENGTH))
    train_neg = train_data.create_dataset("negative", (0, 1944, configurations.ELMO_VECTOR_LENGTH), maxshape=(None, 1944, configurations.ELMO_VECTOR_LENGTH))

    data = read_dataset_data()
    anchor, pos, neg = data[data.columns[0]].to_numpy(), data[data.columns[1]].to_numpy(), data[
        data.columns[2]].to_numpy()
    x_set = np.column_stack((anchor, pos, neg))

    for i in range(0, 5):
        print(i)
        sample = x_set[i]
        anchor_pos = np.array(get_combinations(get_ready_vector(sample[0]), get_ready_vector(sample[1]),
                                                max_text_length=configurations.MAX_TEXT_WORD_LENGTH,
                                                word_embedding_length=configurations.ELMO_VECTOR_LENGTH))
        anchor_neg = np.array(get_combinations(get_ready_vector(sample[0]), get_ready_vector(sample[2]),
                                                max_text_length=configurations.MAX_TEXT_WORD_LENGTH,
                                                word_embedding_length=configurations.ELMO_VECTOR_LENGTH))

        train_pos.resize((train_pos.shape[0] + 1, 1944, configurations.ELMO_VECTOR_LENGTH))
        train_neg.resize((train_neg.shape[0] + 1, 1944, configurations.ELMO_VECTOR_LENGTH))
        train_pos[i] = anchor_pos
        train_neg[i] = anchor_neg
    return 0
'''


