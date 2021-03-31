# %%
# standard
import pandas as pd
import numpy as np
import radiomics
import torch

# fastai
from fastai.tabular import TabularDataBunch, tabular_learner, accuracy, FillMissing, Categorify, Normalize, ClassificationInterpretation, DatasetType, TabularList
from fastai.callbacks import OverSamplingCallback

# personal
from src.utils import get_df_paths, calculate_age, get_df_dis, F_KEY, apply_cat, get_acc, plot_roc_curve, get_advanced_dis_df

# sklearn
from sklearn import tree, metrics
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import class_weight

# display
import matplotlib.pyplot as plt
from IPython.display import Image
from fastai.metrics import roc_curve
from sklearn.metrics import auc


# %% functions

def get_exp_keys(mode, dis, path='radiomics'):
    """remove the keys which have exceptions from the pyradiomics analysis from training"""
    exp = pd.read_csv(f'{path}/{mode}-except.csv')
    exp_keys = exp.keys()
    exp = [exp[key][0] for key in exp_keys][1::]
    
    arr = dis[mode]['idx']

    new_arr = []
    for ele in arr:
        if ele in exp:
            continue
        new_arr.append(ele)
    
    dis[mode]['idx'] = new_arr
    return dis

def gaussianclassifier(df_all, dep_key, trainlen, rounds=20):
    """Gaussian Classifier -> trained for 'rounds' times"""
    # get empty metrics
    auca = []
    acca = []
    precs = []
    sens = []

    # get the features
    features = pd.get_dummies(df_all)
    label = np.array(features[dep_key])
    features = features.drop(dep_key, axis = 1)
    features = np.array(features)

    # get training features(x) and labels (y)
    x = features[:trainlen]
    y = label[:trainlen]

    # define test features and labels
    x_test = features[trainlen::]
    y_test = label[trainlen::]

    # define class weights
    s_w = class_weight.compute_sample_weight('balanced', np.unique(y), y)
    s_wi = [s_w[i] for i in y]

    # train for num rounds
    for _ in range(rounds):
        clf = GaussianNB()

        # Train Classifier
        clf.fit(x, y, sample_weight=s_wi)

        # Get accuracy
        y_pred = clf.predict(x_test)
        p_pred = clf.predict_proba(x_test)[:, 1]
        loc_auc = metrics.roc_auc_score(y_test, p_pred)
        loc_acc = metrics.accuracy_score(y_test, y_pred)
        cm = metrics.confusion_matrix(y_test, y_pred)
        precision = cm[0][0] / (cm[0][0] + cm[0][1])
        sensitifity = cm[1][1] / (cm[1][0] + cm[1][1])

        # append current result to scores
        acca.append(loc_acc)
        auca.append(loc_auc)
        precs.append(precision)
        sens.append(sensitifity)

    # print final result
    print_results(auca, acca, sens, precs)


def randomforestclassifier(df_all, dep_key, trainlen, rounds=20):
    """Random Forest Classifier -> trained for 'rounds' times"""
    # get empty metrics
    auca = []
    acca = []
    precs = []
    sens = []

    # get the features
    features = pd.get_dummies(df_all)
    label = np.array(features[dep_key])
    features= features.drop(dep_key, axis = 1)
    features = np.array(features)

    # get training features(x) and labels (y)
    x = features[:trainlen]
    y = label[:trainlen]

    # define test features and labels
    x_test = features[trainlen::]
    y_test = label[trainlen::]

    # define class weights
    s_w = class_weight.compute_sample_weight('balanced', np.unique(y), y) 
    s_wi = [s_w[i] for i in y]

    for i in range(rounds):
        clf = RandomForestClassifier(n_estimators=200, max_depth=3, random_state=i)

        # Train Classifier
        clf.fit(x, y, sample_weight=s_wi)

        # Get accuracy
        y_pred = clf.predict(x_test)
        p_pred = clf.predict_proba(x_test)[:, 1]
        loc_auc = metrics.roc_auc_score(y_test, p_pred)
        loc_acc = metrics.accuracy_score(y_test, y_pred)
        cm = metrics.confusion_matrix(y_test, y_pred)
        precision = cm[0][0] / (cm[0][0] + cm[0][1])
        sensitifity = cm[1][1] / (cm[1][0] + cm[1][1])

        # append current result to scores
        acca.append(loc_acc)
        auca.append(loc_auc)
        precs.append(precision)
        sens.append(sensitifity)

    # print final result
    print_results(auca, acca, sens, precs)


def fully_connected_learner(data, rounds=20):
    """Neural Network Classifier -> trained for 'rounds' times"""
    auca = []
    acca = []
    precs = []
    sens = []

    for _ in range(rounds):
        # train
        learn = tabular_learner(data, layers=[200,100,100], metrics=accuracy, callback_fns=[OverSamplingCallback])
        learn.fit_one_cycle(10, max_lr=1e-3)
        interp = ClassificationInterpretation.from_learner(learn, DatasetType.Test)
        y_test = interp.y_true

        # Get accuracy
        p_pred = interp.preds[:, 1]
        loc_auc = metrics.roc_auc_score(y_test, p_pred)
        loc_acc = get_acc(interp)
        cm = interp.confusion_matrix()
        precision = cm[0][0] / (cm[0][0] + cm[0][1])
        sensitifity = cm[1][1] / (cm[1][0] + cm[1][1])

        acca.append(loc_acc)
        auca.append(loc_auc)
        precs.append(precision)
        sens.append(sensitifity)
    
    print_results(auca, acca, sens, precs)
    interp = ClassificationInterpretation.from_learner(learn, DatasetType.Test)
    return interp

    

def print_results(auca, acca, sens, spec):
    """print the results of AUC, ACC, Specificity, Sensitivity"""
    print(f'AUC: {round(np.mean(auca), 2)} +/- {round(np.std(auca), 2)}')
    print(f'Accuracy: {round(np.mean(acca), 2)} +/- {round(np.std(acca), 2)}')
    print(f'Sensitivity: {round(np.mean(sens), 2)} +/- {round(np.std(sens), 2)}')
    print(f'Specificity: {round(np.mean(spec), 2)} +/- {round(np.std(spec), 2)}')