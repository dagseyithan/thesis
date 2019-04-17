from keras.layers import Conv1D
from keras.models import Sequential
from keras.layers import Input
from keras import backend as K
import numpy as np


MAX_SENTENCE_LENGTH = 10
KERNEL_SIZE_UNIGRAM = 1
KERNEL_SIZE_BIGRAM = 2
KERNEL_SIZE_TRIGRAM = 3
NUM_FILTERS = 20


def create_model():
    input_unigram = Input(shape=(MAX_SENTENCE_LENGTH, 300))
    unigram_conv = Conv1D(filters=NUM_FILTERS, kernel_size=KERNEL_SIZE_UNIGRAM, kernel_initializer='ones', use_bias=True, activation=None)(input_unigram)
    input_bigram = Input(shape=(MAX_SENTENCE_LENGTH, 300))
    bigram_conv =  Conv1D(filters=NUM_FILTERS, kernel_size=KERNEL_SIZE_BIGRAM, kernel_initializer='ones', use_bias=True, activation=None)(input_bigram)
    input_trigram = Input(shape=(MAX_SENTENCE_LENGTH, 300))
    trigram_conv = Conv1D(filters=NUM_FILTERS, kernel_size=KERNEL_SIZE_TRIGRAM, kernel_initializer='ones', use_bias=True, activation=None)(input_trigram)