import numpy as np

MAX_WORD_CHARACTER_LENGTH = 3

letter = {}
letter['a'] = 0
letter['b'] = 1
letter['c'] = 2




letter_decode = {}
for char, code in letter.items():
    letter_decode[code] = char

def encode_word(word, return_reverse = True):
    word = word[:MAX_WORD_CHARACTER_LENGTH]
    word_r = word[::-1]
    word_matrix = np.zeros((MAX_WORD_CHARACTER_LENGTH, MAX_WORD_CHARACTER_LENGTH))
    word_r_matrix = np.zeros((MAX_WORD_CHARACTER_LENGTH, MAX_WORD_CHARACTER_LENGTH))

    for position, (char, char_r) in enumerate(zip(word, word_r)):
        try:
            word_matrix[letter[char], position] = 1
            word_r_matrix[letter[char_r], position] = 1
        except KeyError:
            word_matrix[letter['a'], position] = 0
            word_r_matrix[letter['a'], position] = 0
    if return_reverse:
        return word_matrix, word_r_matrix
    else:
        return word_matrix

def encode_number(number):
    number_matrix = np.zeros((10, 10))

    for i in range(9, 0, -1):
        d = number % 10
        number = int(np.floor(number / 10))
        number_matrix[d][i] = 1

    return number_matrix

def decode_matrix(m):
    word = []
    for j in range(0, MAX_WORD_CHARACTER_LENGTH):
        for i in range(0, MAX_WORD_CHARACTER_LENGTH):
            if m[i, j] == 1.0:
                word.append(letter_decode[i])
    return ''.join(word)

