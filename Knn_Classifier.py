#Libraries
import pandas as pd 
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from numpy.random import default_rng


class knn_classifier():
    def __init__(self, x , y, perc_test, perc_train, knn):
        self.x_data = x
        self.y_target = y
        self.perc_test = perc_test
        self.perc_train = perc_train
        self.knn = knn
    def shuffling_data(self):
        rnd = default_rng(seed=12)
        permutation = np.arange(0, self.x_data.shape[0])
        rnd.shuffle(permutation)
        self.x_data = self.x_data[permutation]
        self.y_target = self.y_target[permutation]
    def generating_data(self):
        self.n_test = int(self.perc_test*self.x_data.shape[0])
        self.n_train = int(self.perc_train*self.x_data.shape[0])
        self.x_train , self.y_train = self.x_data[:self.n_train], self.y_target[:self.n_train]
        self.x_test, self.y_test = self.x_data[self.n_train:] , self.y_target[self.n_train:]
    def predict_single(self, par):
        distance = []
        for element in self.x_train:
            euc_d = np.sqrt(np.sum(par - element)**2)
            distance.append(euc_d)
            k_neighboors = np.argsort(distance)[:self.knn]
            most_common = np.bincount(self.y_train[k_neighboors], minlength = 3)
            sing_x_pred = np.argmax(most_common)
        return sing_x_pred
    def knn_pred(self):
        self.lis_t =[]
        for elements in self.x_test:
            st = self.predict_single(elements)
            self.lis_t.append(st)
        print(f'Predictions of y_test: {self.lis_t}')
    def pred_score(self):
        count_true = 0
        count_false = 0
        bool = self.lis_t == self.y_test
        score = (self.lis_t == self.y_test).sum() / len(self.y_test)
        for i in bool:
            if i == False:
                count_false+=1
            else:
                count_true+=1
        print(f'Number of Correct predictions:{count_true}.\n Number of wrong predections:{count_false}\n Percent of prediction {score}.')