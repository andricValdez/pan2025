
from sklearn.utils import shuffle
import re
import pandas as pd
import os
import joblib
import glob
import nltk
from nltk.corpus import stopwords
import json

nltk.download('stopwords')


#ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) 
ROOT_DIR = '/home/avaldez/projects/pan-clef2025/'


TASK_1_DIR = ROOT_DIR + 'dataset/pan25-generative-ai-detection-task1-train/'
TASK_2_DIR = ROOT_DIR + 'dataset/pan25-generative-ai-detection-task2-train/'

OUTPUT_DIR_PATH = ROOT_DIR + '/outputs/'
INPUTS_DIR_PATH = ROOT_DIR + '/inputs/'


def save_json(data, file_path):
    with open(file_path, "w") as outfile:
        for element in data:  
            json.dump(element, outfile)  
            outfile.write("\n")  

def save_data(data, file_name, path='', format_file='.pkl', compress=False):
    path_file = OUTPUT_DIR_PATH + file_name + format_file
    joblib.dump(data, path_file, compress=compress)

def load_data(file_name, path='', format_file='.pkl', compress=False):
    path_file = OUTPUT_DIR_PATH + file_name + format_file
    return joblib.load(path_file)

def read_csv(file_path):
  df = pd.read_csv(file_path)
  return df

def read_json(file_path):
  df = pd.read_json(file_path, lines=True)
  return df

def read_jsonl(dir_path):
    return pd.read_json(path_or_buf=dir_path, lines=True)


def delete_dir_files(dir_path):
  files = glob.glob(dir_path + '/*')
  for f in files:
      os.remove(f)

def create_dir(dir_path):
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)

def text_normalize(text):
  # text to lower case
  text = text.lower()
  # remove blank spaces
  text = re.sub(r'\s+', ' ', text).strip()
  # remove html tags
  text = re.compile('<.*?>').sub(r'', text)
  # remove special chars
  text = re.sub('[^A-Za-z0-9]+ ', ' ', text)
  text = re.sub('\W+ ',' ', text)
  text = text.replace('"'," ")
  # remove blank spaces
  text = re.sub(r'\s+', ' ', text).strip()
  # remove stop words
  tokens = nltk.word_tokenize(text)
  without_stopwords = [word for word in tokens if not word.lower().strip() in set(stopwords.words('english'))]
  text = " ".join(without_stopwords)
  return text

def dataset_doc_len(df):
    df['word_len'] = df['text'].str.split().str.len()
    return df['word_len'].min(), df['word_len'].max(), int(df['word_len'].mean())

# PAN 2025 Dataset
def read_pan25_dataset(dataset_dir):
    corpus_train_docs = read_json(file_path=dataset_dir + 'train.jsonl')
    corpus_train_docs = shuffle(corpus_train_docs)

    corpus_val_docs = read_json(file_path=dataset_dir + 'val.jsonl')
    corpus_val_docs = shuffle(corpus_val_docs)

    return corpus_train_docs, corpus_val_docs