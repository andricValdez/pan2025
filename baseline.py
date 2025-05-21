import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score


def get_metrics(predicted, labels, target_names, subset='val'):
    print(f'Accuracy {subset}:', np.mean(predicted == labels))  
    print(f'F1Score {subset}:', f1_score(labels, predicted , average='macro'))
    #print(metrics.classification_report(labels, predicted, target_names=target_names))
    #print(f"Matriz Confusion {subset}: ")
    #print(metrics.confusion_matrix(labels, predicted))

def build_pipeline(model_algo, model_args):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(lowercase=True, analyzer='word', stop_words='english', ngram_range=(1,1))),
        ('clf', CalibratedClassifierCV(model_algo(**model_args))),
    ])
    return pipeline

def train(train_set, model_algo, model_args):
    pipeline = build_pipeline(model_algo, model_args)
    pipeline.fit(train_set['text'], train_set['label'])
    return pipeline

def test(texts, labels, model, target_names, subset='val'):
    predicted = model.predict(texts)
    get_metrics(predicted, labels, target_names, subset=subset)

def baselines(train_set, val_set, test_set, target_names=['human', 'generated'], model='SGDClassifier'):
    print('training model...')
    print("train_set: ", len(train_set))
    print("val_set: ", len(val_set))
    print("test_set: ", len(test_set))
    print('\n')
    model_train = train(train_set, model['algo'], model['args'])   
    test(texts=val_set['text'], labels=val_set['label'], model=model_train, target_names=target_names, subset='val')
    test(texts=test_set['text'], labels=test_set['label'], model=model_train, target_names=target_names, subset='test')
    
def main(train_set, val_set, test_set, models):

    for model in models:
        print(40*'*', 'model: ', model)
        baselines(
            train_set=train_set[ : ], 
            val_set=val_set[ : ], 
            test_set=test_set[ : ],  
            target_names=['human', 'generated'], # human -> 0 | machine -> 1
            model=model,
        )
        print('\n')