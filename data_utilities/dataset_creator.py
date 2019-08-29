import random
from data_utilities.datareader import read_original_products_data, read_dataset_data
import numpy as np
import pandas


def split_dataset(run = False):
    if run:
        data = read_dataset_data('split')
        anchor, pos, neg = data[data.columns[0]], data[data.columns[1]], data[data.columns[2]]
        rand = random.sample(range(0, len(anchor)), 1000)

        with open('dataset_mixed_train.csv','w', encoding='utf-8') as file_train, open('dataset_mixed_test.csv','w', encoding='utf-8') as file_test:
            for i in range(0, len(anchor)):
                if i in rand:
                    file_test.write(anchor[i]+';'+pos[i]+';'+neg[i]+'\n')
                else:
                    file_train.write(anchor[i] + ';' + pos[i] + ';' + neg[i] + '\n')
        file_test.close()
        file_train.close()
    return None


def create_dataset(run = False):
    if run:
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
    return None


def create_soft_margin_dataset(run = False):
    if run:
        Data = read_original_products_data()
        ProductX, ProductY = Data[Data.columns[0]], Data[Data.columns[4]]

        with open('dataset_soft.csv','w', encoding='utf-8') as file:
            for i in range(0, len(ProductX)):
                next = i + 1 if i < len(ProductX) else i - 2
                file.write(ProductX[i]+';'+ProductY[i]+';'+ProductY[next]+'\n')
        file.close()
    return None


def create_mixed_margin_dataset(run = False):
    if run:
        Data = read_original_products_data()
        ProductX, ProductY = Data[Data.columns[0]], Data[Data.columns[4]]

        with open('dataset_mixed.csv','w', encoding='utf-8') as file:
            rand = 0
            for i in range(0, len(ProductX)):
                rand = random.randint(0, len(ProductX) - 1)
                while rand == i:
                    rand = random.randint(0, len(ProductX) - 1)
                next = i + 1 if i + 1 < len(ProductX) else i - 2
                file.write(ProductX[i]+';'+ProductY[i]+';'+ProductY[next]+'\n')
                file.write(ProductX[i]+';'+ ProductY[i]+';'+ProductY[rand]+'\n')
        file.close()
    return None


def create_string_dataset(run = False):
    m = 0
    st_set = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0., 0., 0.]]
    for i in range(0, 4):
        for j in range(0, 4):
            if i == 3:
                j = 3
            for k in range(0, 4):
                if j == 3:
                    k = 3
                arr=np.column_stack((st_set[i], st_set[j], st_set[k]))
                print(m)
                m+=1
                print(arr)

def split_dataset_sick():
    data = pandas.read_csv(r'C:\Users\seyit\PycharmProjects\thesis\data\SICK.txt', header=0, sep='\t',
                           error_bad_lines=False)
    sentences_A, sentences_B, labels = data[data.columns[1]].to_numpy().tolist(), data[data.columns[2]].to_numpy().tolist(), \
                                     data[data.columns[3]].to_numpy().tolist()
    randi = np.random.randint(low=0, high=len(sentences_A), size=1000).tolist()

    sentences_A = [sentences_A[ind] for ind, x in enumerate(sentences_A) if ind not in randi]
    sentences_B = [sentences_B[ind] for ind, x in enumerate(sentences_B) if ind not in randi]
    labels = [labels[ind] for ind, x in enumerate(labels) if ind not in randi]

    sentences_A_t = [sentences_A[ind] for ind, x in enumerate(sentences_A) if ind  in randi]
    sentences_B_t = [sentences_B[ind] for ind, x in enumerate(sentences_B) if ind  in randi]
    labels_t = [labels[ind] for ind, x in enumerate(labels) if ind in randi]

    with open(r'C:\Users\seyit\PycharmProjects\thesis\data\sick_train.txt','w', encoding='utf-8') as file:
        for i in range(0, len(sentences_A)):
            file.writelines(sentences_A[i]+'\t'+sentences_B[i]+'\t'+labels[i]+'\n')
    with open(r'C:\Users\seyit\PycharmProjects\thesis\data\sick_test.txt','w', encoding='utf-8') as file:
        for i in range(0, len(sentences_A_t)):
            file.writelines(sentences_A_t[i]+'\t'+sentences_B_t[i]+'\t'+labels_t[i]+'\n')
    return 1
