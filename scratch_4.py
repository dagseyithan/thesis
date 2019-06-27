import structural
import numpy as np
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
print(structural.get_hybrid_similarity('kopf', 'xxkx'))
print(structural.get_hybrid_similarity('kopf', 'kopf'))
print(structural.get_hybrid_similarity('kopfx', 'kopf'))
print(structural.get_hybrid_similarity('bearbeitung', 'abteilung'))
print(structural.get_hybrid_similarity('kopf', 'schwarzkopf'))
print(structural.get_hybrid_similarity('xschwarzkopf', 'kopf'))