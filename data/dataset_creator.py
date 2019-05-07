import pandas
from config import configurations
import text_utilities as tu
import numpy as np
from data.datareader import readdata
from elmo import __get_elmo_sentence_embedding, __get_elmo_word_embedding



Data = readdata()
ProductX, ProductY = Data[Data.columns[0]], Data[Data.columns[4]]


print(ProductX[0])
text, extracted, numerals = tu.pre_process(ProductX[0])
print(text)
embeddingword = __get_elmo_word_embedding('hallo')
embeddingsent = __get_elmo_sentence_embedding(text)
print(embeddingword.shape)
print(embeddingword)
print(embeddingsent.shape)
print(embeddingsent)

