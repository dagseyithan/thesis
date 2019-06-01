from model_arc2 import get_similarity_arc2, model
from data_utilities.datareader import read_original_products_data
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import pprint as pp
from data_utilities.generator import Native_Test_DataGenerator_for_Arc2
from model_independent import get_similarity_from_independent_model



Data = read_original_products_data()
ProductY = Data[Data.columns[4]].to_numpy()


test = 'Teppich MICHALSKY München anthrazit 133x190 cm'

print(get_similarity_from_independent_model(test, 'Teppich MM München 133x190cm anthrazit'))
print(get_similarity_from_independent_model(test, 'Vliesfotot. 184x248 Brooklyn'))

print(get_similarity_from_independent_model('Teppich MM München 133x190cm anthrazit', test))
print(get_similarity_from_independent_model('Vliesfotot. 184x248 Brooklyn', test))

'''
results = model.predict_generator(generator=Native_Test_DataGenerator_for_Arc2(test), verbose=0, workers=1, use_multiprocessing=False)

indices = np.argsort(-results)

for idx in indices[:10]:
    print(ProductY[idx])
'''


