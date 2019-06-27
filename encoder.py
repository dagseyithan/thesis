import numpy as np
import pprint as pp
from scipy.special import softmax
from scipy.spatial import distance

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





def encode_number(number):
    number_matrix = np.zeros((10, 10))

    for i in range(9, 0, -1):
        d = number % 10
        number = int(np.floor(number / 10))
        number_matrix[d][i] = 1

    return number_matrix


