from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn import neighbors
from sklearn.ensemble import GradientBoostingClassifier
from pyod.models.knn import KNN   # kNN detector
# HBOS: https://pyod.readthedocs.io/en/latest/pyod.models.html#module-pyod.models.hbos
from pyod.models.hbos import HBOS
# IForest: https://pyod.readthedocs.io/en/latest/pyod.models.html#module-pyod.models.iforest
from pyod.models.iforest import IForest
# OCSVM https://pyod.readthedocs.io/en/latest/pyod.models.html#module-pyod.models.ocsvm
from pyod.models.ocsvm import OCSVM
# PCA: https://pyod.readthedocs.io/en/latest/pyod.models.html#module-pyod.models.pca
from pyod.models.pca import PCA
import sys
import os
import numpy as np
import pandas as pd
import importlib
import matplotlib.pyplot as plt
from pandasgui import show
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches
from mxnet import autograd, gluon, init, nd
from sklearn.ensemble import RandomForestClassifier
import ICML_ECAD_helpers as AD_algos
from ICML_ECAD_helpers import get_data_p_val, get_anomalies_classification, find_anomalies, accuracies, PR_curve, ROC_curve

'''Results for Figure 4 by ECAD '''
'''Credit Card Fraud # 2: Data retrieved here: https://www.kaggle.com/mlg-ulb/creditcardfraud
    Data shape is (284807, 31)'''

'''Get data and downsample'''


def get_data_and_true_abnormal(tot_len):
    dataset = pd.read_csv('Money_Laundry.csv', nrows=tot_len)
    dataset.drop('Time', axis=1, inplace=True)
    true_abnormal = np.where(dataset.Class == 1)[0]
    return dataset, true_abnormal


def get_downsample_data(data, train_size):
    data_x = data.iloc[:, :-1]
    data_y = data.iloc[:, -1]
    data_x_numpy = data_x.to_numpy()  # Convert to numpy
    data_y_numpy = data_y.to_numpy()  # Convert to numpy
    X_train = data_x_numpy[:train_size, :]
    X_predict = data_x_numpy[train_size:, :]
    Y_train = data_y_numpy[:train_size]
    Y_predict = data_y_numpy[train_size:]
    train_abnormal = np.where(Y_train == 1)[0]
    train_normal = np.where(Y_train == 0)[0]
    X_train_abnormal = X_train[train_abnormal, :]
    Y_train_abnormal = Y_train[train_abnormal]
    down_sample_idx = np.random.choice(train_normal, 5*len(train_abnormal), replace=False)
    X_train_normal = X_train[down_sample_idx, :]
    Y_train_normal = Y_train[down_sample_idx]
    X_train = np.vstack((X_train_abnormal, X_train_normal))
    Y_train = np.hstack((Y_train_abnormal, Y_train_normal))
    return (X_train, X_predict, Y_train, Y_predict)


'''(For final results) Put everything together (with competing methods)'''

'''First define functions'''
# Competing methods


def mod_to_result(regr_name, X_train, Y_train, test_true_abnormal):
    mod = eval(regr_name)
    mod.fit(X_train, Y_train)
    est_anomalies = mod.predict(X_predict)
    est_anomalies = np.where(est_anomalies == 1)[0]
    precision, recall, F1 = AD_algos.accuracies(
        est_anomalies, test_true_abnormal)
    return [precision, recall, F1]

# ECAD (note, NO refitting after 50% or others)


def ECAD(tot_size, train_frac):
    data, true_abnormal = get_data_and_true_abnormal(tot_size)
    data.shape
    train_size = int(data.shape[0]*train_frac)
    train_size  # A lot of anomalies occurred around 6000
    neighbor_size = 5  # for each abnormal idx in training, how many of its neighbors are used in calibrating residuals
    alpha = np.linspace(0.05, 0.15, 3)  # @ 0.005, ~80/60/70. @ 0.1, almost exactly the same
    alpha = [0.05]
    dotted = True
    stride = 1  # A large stride, suitable when training data is large (so less percentile needed)
    # NOTE: actual getting residual part is the most expensive (since iterate through n...)
    # Thus, let n be even smaller
    return_fitted = False
    est_anomalies = AD_algos.get_anomalies_classification(
        data, 'RF', train_size, alpha,  stride, dotted, return_fitted=return_fitted, neighbor_size=neighbor_size)
    if len(alpha) > 1:
        AD_algos.PR_curve(est_anomalies, true_abnormal[true_abnormal > train_size]-train_size)
    else:
        precision, recall, F1 = AD_algos.accuracies(
            est_anomalies[0], true_abnormal[true_abnormal > train_size]-train_size)
        return [precision, recall, F1]


'''Next experiments'''
tot_trial = 1
train_fracs = np.linspace(0.3, 0.7, 5)
tot_size = 284807
methods = ['ECAD', 'HBOS()', 'IForest()', 'OCSVM()', 'PCA()', 'svm.SVC(gamma="auto")', 'GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)',
           'neighbors.KNeighborsClassifier(n_neighbors=20, weights="distance")', 'MLPClassifier(solver="lbfgs", alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)']
method_name = {'ECAD': 'ECAD', 'HBOS()': 'HBOS', 'IForest()': 'IForest', 'OCSVM()': "OCSVM", 'PCA()': 'PCA', 'svm.SVC(gamma="auto")': 'SVC', 'GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)': 'GBoosting',
               'neighbors.KNeighborsClassifier(n_neighbors=20, weights="distance")': 'KNN', 'MLPClassifier(solver="lbfgs", alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)': 'MLPClassifer'}
results = pd.DataFrame(columns=['itrial', 'train_frac', 'method',
                                'precision', 'recall', 'F1'])
train_fracs = np.linspace(0.6, 0.7, 2)
for itrial in range(tot_trial):
    np.random.seed(98765+itrial)
    for train_frac in train_fracs:
        train_frac = np.round(train_frac, 2)
        data, true_abnormal = get_data_and_true_abnormal(tot_size)
        train_size = int(tot_size*train_frac)
        X_train, X_predict, Y_train, _ = get_downsample_data(data, train_size)
        test_true_abnormal = true_abnormal[true_abnormal > train_size]-train_size
        for method in methods:
            if method == 'ECAD':
                precision, recall, F1 = ECAD(tot_size, train_frac)
            else:
                precision, recall, F1 = mod_to_result(method, X_train, Y_train, test_true_abnormal)
            results.loc[len(results)] = [itrial, train_frac,
                                         method_name[method], precision, recall, F1]
            results.to_csv(f'Kaggle_results.csv', index=False)


results = pd.read_csv('Kaggle_results.csv')
AD_algos.plt_prec_recall_F1(results)
