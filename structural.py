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
    return total/count

def get_encoded_similarity(a, b, n = 3):
    a_r = np.flip(a, axis=1)
    b_r = np.flip(b, axis=1)

    if a.any():
        while not a_r[:,0:1].any():
            a_r = np.roll(a_r, -1, axis=1)
    if b.any():
        while not b_r[:,0:1].any():
            b_r = np.roll(b_r, -1, axis=1)

    '''
    #print(a)
    #print(a_r)
    #print(b)
    #print(b_r)

    len_a = a.sum()
    len_b = b.sum()
    max_len = len_a if len_a > len_b else len_b
    dist, dist_t, dist_r, dist_t_r = 0.0, 0.0, 0.0, 0.0
    i = 0
    for coli_at, coli_bt, coli_at_r, coli_bt_r in zip(a.transpose(), b.transpose(), a_r.transpose(), b_r.transpose()):
        if i >= max_len:
            break
        dist_t += np.multiply(coli_at, coli_bt).sum()
        dist_t_r += np.multiply(coli_at_r, coli_bt_r).sum()
        i+=1
    i = 0
    for rowi_a, rowi_b in zip(a, b):
        if rowi_a.any() and rowi_b.any():
            dist += (hamming(rowi_a, rowi_b))
            i+=1
        elif (rowi_a.any() and not rowi_b.any()) or (not rowi_a.any() and rowi_b.any()):
            i+=1
    for rowi_a, rowi_b in zip(a_r, b_r):
        if rowi_a.any() and rowi_b.any():
            dist_r +=  (hamming(rowi_a, rowi_b))
            i+=1
        elif (rowi_a.any() and not rowi_b.any()) or (not rowi_a.any() and rowi_b.any()):
            i+=1
    #print(max_len)
    #print('dist: ' + str(dist))
    #print('dist_r: ' + str(dist_r))
    #print('dist_t: ' + str(dist_t))
    #print('dist_t_r: ' + str(dist_t_r))



    dist = dist /max_len
    dist_r = dist_r/max_len
    dist_t = dist_t /max_len
    dist_t_r = dist_t_r/max_len
    #print('encoded_dist:')
    #print(((dist_t + dist)* 1./2.0) + ((dist_r + dist_t_r) * 1./2.0))

    op_hadamard = np.multiply(a, b).sum()
    op_r_hadamard = np.multiply(a_r, b_r).sum()
    op = (a.sum() + b.sum())

    #print(op_hadamard)
    #print(op_r_hadamard)

    #print('encoded + encoded_dist:')
    #print()

    hadamard = ((op_hadamard+op_r_hadamard)/op) if op != 0.0 else 0.0
    encoded = (((dist_t + dist)* 1./2.0) + ((dist_r + dist_t_r) * 1./2.0))

    total = (hadamard + encoded)*0.5

    return total
    '''
    op_hadamard = np.multiply(a, b).sum()
    op_r_hadamard = np.multiply(a_r, b_r).sum()
    op = (a.sum() + b.sum())
    return ((op_hadamard + op_r_hadamard) + np.abs(op_hadamard - op_r_hadamard)) / op if op != 0.0 else 0.0


def get_mean_convolutional_similarity(worda, wordb):
    print('n=2:')
    print(get_convolutional_similarity(worda, wordb, n=2, stride=2))
    print(get_convolutional_similarity(worda, wordb, n=3, stride=3))
    print(get_convolutional_similarity(worda, wordb, n=4, stride=4))

    return 1#get_convolutional_similarity(worda, wordb, n=3, stride=1)


print(get_mean_convolutional_similarity('bearbeitung', 'ableitung'))
print(get_mean_convolutional_similarity('rel   ', 'relhok'))

#get_convolutional_similarity('aba', 'aba', n=2, stride=2)