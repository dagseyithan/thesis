import structural_regular
import structural3x3
import encoder3x3
import numpy as np
from scipy.linalg import norm
from openeye import oegraphsim
import textdistance
from nltk.metrics.distance import edit_distance
from scipy.spatial.distance import hamming, euclidean,cosine, minkowski, cityblock, jaccard, chebyshev, sokalsneath, yule, russellrao, dice, kulsinski, braycurtis, sokalmichener
from scipy.stats import entropy, wasserstein_distance, energy_distance, describe
#print(str(minkowski([1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1], [1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0])))
#print(str(sokalmichener([1,0,0,0,0,1], [0,0,0,1,1,1])))
#print(describe([0,1,0,0,0]))
#print(describe([1,0,0,0,0]))
#print(textdistance.tanimoto('kopf', 'kopfx'))
#print(structural.get_hybrid_similarity('cabbab', 'cacaac'))
#print('------')
#print(structural3x3.get_hybrid_similarity('bba', 'cca'))
#print('-------')
#print(structural3x3.get_hybrid_similarity('bab', 'aac'))

from pyjarowinkler.distance import get_jaro_distance
from nltk.metrics.distance import edit_distance
from scipy.spatial.distance import sokalmichener, hamming
'''
print(structural.get_hybrid_similarity('kopf', 'kopf'))
print(structural.get_hybrid_similarity('kopfx', 'kopf'))
print(structural.get_hybrid_similarity('bearbeitung', 'abteilung'))
print(structural.get_hybrid_similarity('kopf', 'schwarzkopf'))
print(structural.get_hybrid_similarity('xschwarzkopf', 'kopf'))
'''

print(get_jaro_distance('1110110101', '11100000'))
print(edit_distance('1110110101', '11100000'))
print(hamming('1110110101', '11100000'))
'''



for a, a_r in generate_strings():
    for b, b_r in generate_strings():
        op_hadamard = np.multiply(a, b).sum()
        op_r_hadamard = np.multiply(a_r, b_r).sum()
        op = (a.sum() + b.sum())

        dist = (op_hadamard+op_r_hadamard)/op
        #no = 1.0 - (norm(np.subtract(a, b)) / np.add(a, b).sum())
        no = 1.0 - ((np.linalg.norm(np.subtract(a, b)) + np.linalg.norm(np.subtract(a_r, b_r))) / np.add(a, b).sum())

        print(encoder3x3.decode_matrix(a))
        print(encoder3x3.decode_matrix(b))
        print(dist)
        print(no)
        print((dist+no)/2.0)
'''

def generate_strings():
    st_set=[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0., 0., 0.]]
    for i in range(0, 4):
        for j in range(0, 4):
            for k in range(0, 4):
                arr=np.column_stack((st_set[i], st_set[j], st_set[k]))
                arr_r = np.flip(arr, axis=1)
                print(arr)

generate_strings()