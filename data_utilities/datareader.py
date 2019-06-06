import pandas
from config import configurations
import platform


if platform.system() == 'Linux':
    original_file = configurations.LINUX_ORIGINAL_PRODUCTS_FILE_PATH
    dataset_file = configurations.LINUX_DATASET_FILE_PATH
else:
    original_file = configurations.WINDOWS_ORIGINAL_PRODUCTS_FILE_PATH
    dataset_file = configurations.WINDOWS_DATASET_FILE_PATH


def read_original_products_data():
    return pandas.read_csv(original_file, header=1, encoding='utf-8')

def read_dataset_data(mode = 'train'):
    if mode == 'train':
        return pandas.read_csv(dataset_file + 'dataset_mixed_train.csv', header=0, sep=';', encoding='utf-8')
    elif mode == 'test':
        return pandas.read_csv(dataset_file + 'dataset_test.csv', header=0, sep=';', encoding='utf-8')
    elif mode == 'split':
        return pandas.read_csv(dataset_file + 'dataset_mixed.csv', header=0, sep=';', encoding='utf-8')