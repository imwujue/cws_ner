import pickle
import pandas as pd
import re
import numpy as np
import codecs
import dataset
import itertools
from sklearn import metrics

def evaluate():
    file_true = "data/testset1/test_cws1.txt"
    file_true_tags = 'data/process/test_cws_tags.txt'
    file_pred = "data/process/test_res.txt"
    file_pred_tags = 'data/process/test_pred_tags.txt'
    dataset.addTags(file_true, file_true_tags)
    dataset.addTags(file_pred, file_pred_tags)
    _, label_true = dataset.generateLabel(file_true_tags)
    _, label_pred = dataset.generateLabel(file_pred_tags)
    label_true = list(itertools.chain.from_iterable(label_true))
    # print(label_true)
    label_pred = list(itertools.chain.from_iterable(label_pred))
    precision_score_class = metrics.precision_score(label_true, label_pred, labels=['S','B','M','E'], average = None)
    recall_score_class = metrics.recall_score(label_true, label_pred, labels=['S','B','M','E'], average = None)
    f1_score_class = metrics.f1_score(label_true, label_pred, labels=['S','B','M','E'], average= None)
    precision_score = metrics.precision_score(label_true, label_pred, labels=['S', 'B', 'M', 'E'], average='weighted')
    recall_score = metrics.recall_score(label_true, label_pred, labels=['S', 'B', 'M', 'E'], average='weighted')
    f1_score = metrics.f1_score(label_true, label_pred, labels=['S', 'B', 'M', 'E'], average='weighted')

    print("Evaluate Result of Model:")
    print("pression_score: ['S', 'B', 'M', 'E']")
    print(precision_score_class)
    print(precision_score, end="\n\n")
    print("recall_score: ['S', 'B', 'M', 'E']")
    print(recall_score_class)
    print(recall_score, end="\n\n")
    print("f1_score: ['S', 'B', 'M', 'E']")
    print(f1_score_class)
    print(f1_score, end="\n\n")
    return precision_score, recall_score, f1_score
