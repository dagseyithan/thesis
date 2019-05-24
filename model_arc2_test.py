from model_arc2 import get_similarity_arc2, model
from data_utilities.datareader import read_original_products_data
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import pprint as pp
from data_utilities.generator import Native_Test_DataGenerator_for_Arc2



Data = read_original_products_data()
ProductY = Data[Data.columns[4]].to_numpy()


test = 'Teppich MICHALSKY München anthrazit 133x190 cm'


results = model.predict_generator(generator=Native_Test_DataGenerator_for_Arc2(test), verbose=0, workers=16, use_multiprocessing=False)


pp.pprint(results)


