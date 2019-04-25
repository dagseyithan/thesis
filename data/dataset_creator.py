import pandas
from config import configurations
import text_utilities as tu
import numpy as np
import platform


if platform.system() == 'Linux':
    file = configurations.LINUX_ORIGINAL_PRODUCTS_FILE_PATH
else:
    file = configurations.WINDOWS_ORIGINAL_PRODUCTS_FILE_PATH


Data = pandas.read_csv(file, header=1)
ProductX, ProductY = Data[Data.columns[0]], Data[Data.columns[4]]


print(ProductX[0])
text, extracted, numerals = tu.pre_process(ProductX[0])
print(text)
print(extracted)
print(numerals)
