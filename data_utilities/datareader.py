import pandas
from config import configurations
import platform


if platform.system() == 'Linux':
    original_file = configurations.LINUX_ORIGINAL_PRODUCTS_FILE_PATH
    dataset_file_path = configurations.LINUX_DATASET_FILE_PATH
else:
    original_file = configurations.WINDOWS_ORIGINAL_PRODUCTS_FILE_PATH
    dataset_file_path = configurations.WINDOWS_DATASET_FILE_PATH


def read_original_products_data():
    return pandas.read_csv(original_file, header=1, encoding='utf-8')

def read_dataset_data(mode = 'train'):
    if mode == 'train':
        return pandas.read_csv(dataset_file_path + 'dataset_mixed_train.csv', header=0, sep=';', encoding='utf-8')
    elif mode == 'test':
        return pandas.read_csv(dataset_file_path + 'dataset_test.csv', header=0, sep=';', encoding='utf-8')
    elif mode == 'split':
        return pandas.read_csv(dataset_file_path + 'dataset_mixed.csv', header=0, sep=';', encoding='utf-8')

def read_german_words_dictionary():
    with open('C:\\Users\\seyit\\PycharmProjects\\thesis\\data\\german_words_dictionary\\german.txt','r', encoding='latin1') as file:
        dic = [str(word).lower().strip() for word in file]
    file.close()
    return dic


def read_sts_data(mode = 'train'):
    if mode == 'train':
        data = pandas.read_csv(dataset_file_path + 'sts_train.csv', header=None, sep='\t', error_bad_lines=False)
    elif mode == 'test':
        data = pandas.read_csv(dataset_file_path + 'sts_test.csv', header=None, sep='\t', error_bad_lines=False,
                            engine='python')
    elif mode == 'split':
        data = pandas.read_csv(dataset_file_path + 'sts_mixed.csv', header=0, sep='\t', error_bad_lines=False)

    sentence_A, sentence_B, scores = data[data.columns[5]].to_numpy(), data[data.columns[6]].to_numpy(), \
                                     data[data.columns[4]].to_numpy()
    return sentence_A, sentence_B, scores

