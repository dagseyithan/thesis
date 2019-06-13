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


word_matrix = np.zeros((30, 30))

word = 'schule'


for position, char in enumerate(word):
    word_matrix[letter[char], position] = 1

m = np.ones((2, 2))
print(softmax(m))

print(distance.euclidean([1, 0, 1], [0, 1, 0]))

print(distance.euclidean([0, 0, 0], [0, 0, 1]))

print(distance.euclidean([0, 1, 0], [1, 0, 0]))

print('\n\n')
sim = (distance.euclidean([1, 0, 1], [0, 1, 0]) + distance.euclidean([0, 0, 0], [0, 0, 1]) + distance.euclidean([0, 1, 0], [1, 0, 0]))/3.0
print(np.log(sim))
