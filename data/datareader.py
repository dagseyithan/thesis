import pandas
from config import configurations
import platform


if platform.system() == 'Linux':
    file = configurations.LINUX_ORIGINAL_PRODUCTS_FILE_PATH
else:
    file = configurations.WINDOWS_ORIGINAL_PRODUCTS_FILE_PATH


Data = pandas.read_csv(file, header=1)


def readdata():
    return Data