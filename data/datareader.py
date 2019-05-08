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

def read_dataset_data():
    return pandas.read_csv(dataset_file, header=1, encoding='utf-8')