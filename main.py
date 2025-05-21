import pandas as pd
import numpy as np
import random
import json
import random
import glob
import pprint
import os
import joblib
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

import utils
import baseline


def main2():
    corpus_train_docs, corpus_test_docs = utils.read_pan25_dataset(dataset_dir=utils.TASK_1_DIR)

    # Split data into 80% train and 20% test, stratifying by the 'label' column
    corpus_train_docs, corpus_val_docs = train_test_split(corpus_train_docs, test_size=0.2, random_state=42, stratify=corpus_train_docs['label'])

    print("************************************ TRAIN SET: ")
    print(corpus_train_docs.info())
    print("************************************ VAL SET: ")
    print(corpus_val_docs.info())
    print("************************************ VAL SET: ")
    print(corpus_test_docs.info())

    print("************************************ BASELINES: ")
    #'LinearSVC', 'MultinomialNB', 'LogisticRegression', 'xgb.XGBClassifier', 'SGDClassifier'
    models = [
        {"algo": LinearSVC, "args": {"dual": "auto","random_state": 42}},
        {"algo": LogisticRegression, "args": {}},
        {"algo": SGDClassifier, "args": {}},
    ]
    baseline.main(corpus_train_docs, corpus_val_docs, corpus_test_docs, models)



def build_pipeline(model):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(lowercase=True, analyzer='word', stop_words='english', ngram_range=(1,1))),
        ('clf', CalibratedClassifierCV(model())),
    ])
    return pipeline


def main(args):
    #llm_baseline()
    #return

    #print(known_args)
    algo_ml = 'SGDClassifier' # SGDClassifier, LinearSVC, MultinomialNB, LogisticRegression
    model_name = 'Model_A_clf_train_' # Model_A_clf_train_, pan24_smoke_train_
    train_set_name = 'Partition_A_train_set' # Partition_A_train_set, pan24_generative_authorship_smoke_train
    test_set_name  = ''  #Partition_A_test.jsonl

    if args.exec_type == 'train':
        #***** read data
        train_set, test_set = utils.read_pan25_dataset(dataset_dir=utils.TASK_1_DIR)
        train_set, val_set = train_test_split(train_set, test_size=0.2, random_state=42, stratify=train_set['label'])

        print("************************************ TRAIN SET: ")
        print(train_set.info())
        print("************************************ VAL SET: ")
        print(val_set.info())
        print("************************************ VAL SET: ")
        print(test_set.info())

        # use only for pan24_smoke_train
        if train_set_name == 'pan24_generative_authorship_smoke_train':
            train_set_tmp_txt1 = pd.DataFrame({'id': train_set['id'], 'text': train_set['text1'], 'class': train_set['label_txt1']})
            train_set_tmp_txt2 = pd.DataFrame({'id': train_set['id'], 'text': train_set['text2'], 'class': train_set['label_txt2']})
            train_set = pd.concat([train_set_tmp_txt1, train_set_tmp_txt2])
            print(train_set.info())

        #***** build baseline model
        clf_models = {
            'LinearSVC': LinearSVC,
            'MultinomialNB': MultinomialNB,
            'LogisticRegression': LogisticRegression,
            'SGDClassifier': SGDClassifier
        }
        
        #***** train model
        print('training model...')
        pipeline = build_pipeline(model = clf_models[algo_ml])
    
        pipeline.fit(train_set['text'], train_set['label'])
        utils.save_data(data=pipeline, file_name=model_name + algo_ml)
        print('Done!')

    else:
        # predictions
        pipeline = utils.load_data(file_name=model_name + algo_ml)
        train_set, test_set = utils.read_pan25_dataset(dataset_dir=utils.TASK_1_DIR)
        #train_set, val_set = train_test_split(train_set, test_size=0.2, random_state=42, stratify=train_set['label'])
        
        cutoff = None
        test_set = test_set[:cutoff]
        y_true = test_set['label']
        test_set = test_set.to_dict('records')

        predictions = []
        y_pred = []
        epsilon = 0.05  # undecidable margin

        for row in test_set:
            probs = np.array([pipeline.predict_proba([row['text']])[0]])
            max_prob = np.max(probs, axis=1)[0]
            pred_class = np.argmax(probs, axis=1)[0]
            y_pred.append(pred_class)

            # Decide the label value
            if abs(max_prob - 0.5) <= epsilon:
                label = 0.5
            elif pred_class == 1:
                label = round(probs[0][1], 4)  # machine
            else:
                label = round(probs[0][1], 4)  # still return class 1 prob

            res = {"id": row["id"], "label": label}
            predictions.append(res)

        utils.save_json(data=predictions, file_path=utils.OUTPUT_DIR_PATH + '/' + model_name + "preds.jsonl")
        
        # *************** Eval  
        print('Accuracy:', accuracy_score(y_true, y_pred))  
        print('F1Score:', f1_score(y_true, y_pred, average='macro'))  



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exec_type", "-t", help="train or test exection ", default='train', type=str)
    parser.add_argument("-in", "--input_dataset", help="input test data ", default='', type=str)
    parser.add_argument("-out", "--onput_dir", help="output dir to save pred data", default='', type=str)
    parser.add_argument("-model", "--model", help="model to train or test ", default='SGDClassifier', type=str)

    args = parser.parse_args()
    print(args)
    #args = vars(args)
    main(args)
    