import numpy as np
import sys
from config import configurations
np.set_printoptions(threshold=sys.maxsize)
MAX_WORD_CHARACTER_LENGTH = configurations.MAX_WORD_CHARACTER_LENGTH
ALPHABET_LENGTH = configurations.ALPHABET_LENGTH

letter = {}
letter['a'] = 0
letter['b'] = 1
letter['c'] = 2
letter['d'] = 3
letter['e'] = 4
letter['f'] = 5
letter['g'] = 6
letter['h'] = 7
letter['i'] = 8
letter['j'] = 9
letter['k'] = 10
letter['l'] = 11
letter['m'] = 12
letter['n'] = 13
letter['o'] = 14
letter['p'] = 15
letter['q'] = 16
letter['r'] = 17
letter['s'] = 18
letter['t'] = 19
letter['u'] = 20
letter['v'] = 21
letter['w'] = 22
letter['x'] = 23
letter['y'] = 24
letter['z'] = 25
letter['ä'] = 26
letter['ö'] = 27
letter['ü'] = 28
letter['ß'] = 29
letter['é'] = 30
letter['&'] = 31
letter['‘'] = 32
letter['-'] = 33
letter[','] = 34
letter['@'] = 35
letter['$'] = 36
letter['!'] = 37
letter['?'] = 38
letter['.'] = 39
letter['_'] = 40
letter['*'] = 41
letter[';'] = 42
letter['#'] = 43
letter['0'] = 44
letter['1'] = 45
letter['2'] = 46
letter['3'] = 47
letter['4'] = 48
letter['5'] = 49
letter['6'] = 50
letter['7'] = 51
letter['8'] = 52
letter['9'] = 53


letter_decode = {}
for char, code in letter.items():
    letter_decode[code] = char

def encode_word(word, return_reverse = True):
    word = word[:MAX_WORD_CHARACTER_LENGTH]
    word_r = word[::-1]
    word_matrix = np.zeros((ALPHABET_LENGTH, MAX_WORD_CHARACTER_LENGTH))
    word_r_matrix = np.zeros((ALPHABET_LENGTH, MAX_WORD_CHARACTER_LENGTH))

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
    for j in range(0, ALPHABET_LENGTH):
        for i in range(0, MAX_WORD_CHARACTER_LENGTH):
            if m[i, j] == 1.0:
                word.append(letter_decode[i])
    return ''.join(word)

def convert_to_tensor(matrix, dim=3):
    tensor = []
    nonzero_mask = []
    for i in range(0, int(ALPHABET_LENGTH), dim):
        for j in range(0, int(MAX_WORD_CHARACTER_LENGTH), dim):
            mat = matrix[i:i+dim, j:j+dim]
            nonzero_mask.append(1 if mat.any() else 0)
            tensor.append(mat)
    return np.array(tensor), np.array(nonzero_mask)


