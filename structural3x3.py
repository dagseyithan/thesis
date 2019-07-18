import math
import numpy as np
import difflib
from scipy import spatial
from scipy.linalg import norm
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import sokalmichener, hamming
from nltk.metrics.distance import edit_distance, masi_distance, jaccard_distance
from ngram import NGram
import encoder3x3
from pyjarowinkler.distance import get_jaro_distance

MAX_WORD_LENGTH = 3.0


def get_encoded_simple_similarity(worda, wordb):
    a, a_r = encoder3x3.encode_word(worda)
    b, b_r = encoder3x3.encode_word(wordb)
    op_hadamard = np.multiply(a, b).sum()
    op_r_hadamard = np.multiply(a_r, b_r).sum()
    op = (a.sum() + b.sum())
    return ((op_hadamard+op_r_hadamard) + np.abs(op_hadamard - op_r_hadamard))/op


def get_encoded_norm_similarity(worda, wordb):
    a, a_r = encoder3x3.encode_word(worda)
    b, b_r = encoder3x3.encode_word(wordb)

    print(b_r)

    len_a = a.sum()
    len_b = b.sum()
    max_len = len_a if len_a > len_b else len_b

    print(np.subtract(a, b))
    return 1.0 - (norm(np.subtract(a, b)) / np.add(a, b).sum())


def get_encoded_similarity(worda, wordb):
    a, a_r = encoder3x3.encode_word(worda)
    b, b_r = encoder3x3.encode_word(wordb)

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
    print(max_len)
    print('dist: ' + str(dist))
    print('dist_r: ' + str(dist_r))
    print('dist_t: ' + str(dist_t))
    print('dist_t_r: ' + str(dist_t_r))



    dist = dist /max_len
    dist_r = dist_r/max_len
    dist_t = dist_t /max_len
    dist_t_r = dist_t_r/max_len
    print('encoded_dist:')
    print(((dist_t + dist)* 1./2.0) + ((dist_r + dist_t_r) * 1./2.0))

    op_hadamard = np.multiply(a, b).sum()
    op_r_hadamard = np.multiply(a_r, b_r).sum()
    op = (a.sum() + b.sum())

    #print(op_hadamard)
    #print(op_r_hadamard)

    print('encoded + encoded_dist:')
    print((((op_hadamard+op_r_hadamard)/op) + (((dist_t + dist)* 1./2.0) + ((dist_r + dist_t_r) * 1./2.0)))*0.5)

    return  ((op_hadamard+op_r_hadamard)/op) if op is not 0.0 else 1.0


def get_ngram_similarity(worda, wordb, n = 2):
    return NGram.compare(worda, wordb, k = n)


def get_edit_distance(worda, wordb):
    return edit_distance(worda, wordb)


def get_alignment_cost_matrix(worda, wordb):
    count = 0
    collection = []
    cost = []
    for ci in worda:
        for cj in wordb:
            collection.append(1.0 if ci is cj else 0.0)
            count = count + 1
        cost.append(collection)
        collection = []
    return count, np.nan_to_num(np.array(cost))


def get_hungarian_alignment_distance(texta, textb):
    count, Matrix = get_alignment_cost_matrix(texta, textb)
    row_ind, col_ind = linear_sum_assignment(Matrix)
    return Matrix[row_ind, col_ind].sum() / count


def get_hybrid_similarity(worda, wordb):
    a, a_r = encoder3x3.encode_word(worda)
    b, b_r = encoder3x3.encode_word(wordb)

    len_a = a.sum()
    len_b = b.sum()
    max_len = len_a if len_a > len_b else len_b

    print('jaro: ' + str(get_jaro_distance(worda, wordb)))
    print('norm: ' + str(get_encoded_norm_similarity(worda, wordb)))
    print('encoded: ' + str(get_encoded_similarity(worda, wordb)))
    print('edit: ' + str(1 - get_edit_distance(worda, wordb)/max_len))
    print('hamming: ' + str(1 - hamming(worda, wordb)/max_len))

    return 0.50 * get_encoded_norm_similarity(worda, wordb) \
         + 0.50 * get_encoded_similarity(worda, wordb)

def get_hybrid_similarity_new(worda, wordb):
    a, a_r = encoder3x3.encode_word(worda)
    b, b_r = encoder3x3.encode_word(wordb)

    print(encoder3x3.decode_matrix(a))
    print(a)
    print(encoder3x3.decode_matrix(b))
    print(b)

    a_flat = np.ndarray.flatten(a)
    b_flat = np.ndarray.flatten(b)

    worda_str = str(np.ndarray.flatten(a))
    wordb_str = str(np.ndarray.flatten(b))
    #print(get_edit_distance(worda_str, wordb_str))


    #print(worda_str)
    #print(wordb_str)

    len_a = a.sum()
    len_b = b.sum()
    max_len = len(worda_str) if len(worda_str) > len(wordb_str) else len(wordb_str)

    #print('jaro:')
    #print(get_jaro_distance('110001000', '1'))

    #print('hamming:')
    #print(hamming(np.ndarray.flatten(a), np.ndarray.flatten(b)))

    #print(len(worda_str) + len(wordb_str))

    #print('jaro: ' + str(get_jaro_distance(worda_str, wordb_str)))
    #print('norm: ' + str(get_encoded_norm_similarity(worda, wordb)))
    print('encoded: ' + str(get_encoded_similarity(worda, wordb)))
    #print('edit: ' + str(1 - get_edit_distance(worda_str, wordb_str)/max_len))
    #print('hamming: ' + str(1 - hamming(a_flat, b_flat)))

    return  0.33 * get_encoded_norm_similarity(worda, wordb) \
         + 0.33 * get_encoded_similarity(worda, wordb) \
         + 0.33 * (1 - hamming(a_flat, b_flat))


print(get_hybrid_similarity_new('', 'a'))
