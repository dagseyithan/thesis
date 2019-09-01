import encoder
from scipy.spatial.distance import hamming
import numpy as np
import pprint as pp


def get_convolutional_similarity(worda, wordb, n = 2, stride = 2):
    count = 0.0
    total = 0.0
    word_matrix_a = encoder.encode_word(worda, return_reverse=False)
    word_matrix_b = encoder.encode_word(wordb, return_reverse=False)

    #pp.pprint(word_matrix_a)
    #pp.pprint(word_matrix_b)

    for i in range(0, encoder.MAX_WORD_CHARACTER_LENGTH, stride):
        for j in range(0, encoder.ALPHABET_LENGTH, stride):
            if i+n >= encoder.MAX_WORD_CHARACTER_LENGTH or j+n >= encoder.ALPHABET_LENGTH:
                break
            sub_matrix_a = word_matrix_a[i:i+n, j:j+n]
            sub_matrix_b = word_matrix_b[i:i+n, j:j+n]

            score = get_encoded_similarity(sub_matrix_a, sub_matrix_b, n)
            #print(score)
            if sub_matrix_a.any() or sub_matrix_b.any():
                count = count + 1.0
                total+= score
    #print(total)
    #print(count)
    return total/count if count is not 0.0 else 0.0

def get_encoded_similarity(a, b, n = 3):

    #a, a_r = encoder.encode_word(a, return_reverse=True)
    #b, b_r = encoder.encode_word(b, return_reverse=True)
    a_r = np.flip(a, axis=1)
    b_r = np.flip(b, axis=1)

    if a.any():
        while not a_r[:,0:1].any():
            a_r = np.roll(a_r, -1, axis=1)
    if b.any():
        while not b_r[:,0:1].any():
            b_r = np.roll(b_r, -1, axis=1)

    op_hadamard = np.multiply(a, b).sum()

    op_r_hadamard = np.multiply(a_r, b_r).sum()

    op = (a.sum() + b.sum())

    return ((op_hadamard + op_r_hadamard) + np.abs(op_hadamard - op_r_hadamard)) / (op + 0.0000001)


def get_mean_convolutional_similarity(worda, wordb):
    return (get_convolutional_similarity(worda, wordb, n=3, stride=3) \
           + get_convolutional_similarity(worda[::-1], wordb[::-1], n=3, stride=3)) * 0.5

'''
a = get_mean_convolutional_similarity('bearbeitun', 'bearbeitung')
print(a)
#print(get_mean_convolutional_similarity('Awomanisdicingsomepeeledpotatoescutintothickstrips', 'Awomanischoppingapeeledpotatointoslices'))
#print(get_mean_convolutional_similarity('bearbeitung', 'bebeitung'))
#print(get_mean_convolutional_similarity('gnutiebraeb', 'gnutiebeb'))
#print(get_mean_convolutional_similarity('rel   ', 'relhok'))


print(get_encoded_similarity('bearbeitung', 'ableitung'))
print(get_encoded_similarity('abteilung', 'ableitung'))
print(get_encoded_similarity('gesellschaft', 'freundschaft'))
print(get_encoded_similarity('gesellschaft', 'ffreundschaft'))
print(get_encoded_similarity('gesellschaft', 'fffreundschaft'))
print(get_encoded_similarity('gesellschaft', 'fffreundschaftt'))
print(get_encoded_similarity('gesellschaft', 'fffreundschafttt'))
print(get_encoded_similarity('freundschaft', 'ffreundschafttt'))
print(get_encoded_similarity('freundschaft', 'ffreund'))
print(get_encoded_similarity('freundschaft', 'schaft'))
print(get_encoded_similarity('freundschaft', 'shcaft'))
print(get_encoded_similarity('freundschaft', 'freund'))
print(get_encoded_similarity('freund', 'freund'))

#get_convolutional_similarity('aba', 'aba', n=2, stride=2)
'''