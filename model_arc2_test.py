from model_arc2 import get_similarity_arc2
from data_utilities.datareader import read_original_products_data
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import pprint as pp



Data = read_original_products_data()
ProductY = Data[Data.columns[4]].to_numpy()


test = 'Teppich MICHALSKY München anthrazit 133x190 cm'
results = []

for product in tqdm(ProductY):
    if len(product.split()) > 1:
        results.append((test, product))

with Pool() as pool:
    pool.starmap(get_similarity_arc2, results)

results = np.array(results)


