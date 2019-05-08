import random
from data.datareader import read_original_products_data




Data = read_original_products_data()
ProductX, ProductY = Data[Data.columns[0]], Data[Data.columns[4]]

with open('dataset.csv','w', encoding='utf-8') as file:
    rand = 0
    for i in range(0, len(ProductX)):
        rand = random.randint(0, len(ProductX) - 1)
        while rand == i:
            rand = random.randint(0, len(ProductX) - 1)
        file.write(ProductX[i]+';'+ProductY[i]+';'+ProductY[rand]+'\n')
file.close()


