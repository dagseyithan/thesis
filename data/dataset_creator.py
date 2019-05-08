import pandas
from config import configurations
import text_utilities as tu
import numpy as np
import random
from data.datareader import readdata
from elmo import __get_elmo_sentence_embedding, __get_elmo_word_embedding



Data = readdata()
ProductX, ProductY = Data[Data.columns[0]], Data[Data.columns[4]]

with open('dataset.csv','w', encoding='utf-8') as file:
    rand = 0
    for i in range(0, len(ProductX)):
        rand = random.randint(0, len(ProductX) - 1)
        while rand == i:
            rand = random.randint(0, len(ProductX) - 1)
        file.write(ProductX[i]+';'+ProductY[i]+';'+ProductY[rand]+'\n')
file.close()


