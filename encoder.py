import numpy as np

LENGTH = 30

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


letter_decode = {}
for char, code in letter.items():
    letter_decode[code] = char

def encode_word(word, return_reverse = False):
    word = word[:30]
    word_r = word[::-1]
    word_matrix = np.zeros((LENGTH, LENGTH))
    word_r_matrix = np.zeros((LENGTH, LENGTH))

    for position, (char, char_r) in enumerate(zip(word, word_r)):
        try:
            word_matrix[letter[char], position] = 1
            word_r_matrix[letter[char_r], position] = 1
        except KeyError:
            word_matrix[letter['a'], position] = 1
            word_r_matrix[letter['a'], position] = 1
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
    for j in range(0, LENGTH):
        for i in range(0, LENGTH):
            if m[i, j] == 1.0:
                word.append(letter_decode[i])
    return ''.join(word)

