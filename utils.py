
from sklearn.utils import shuffle
import re
import pandas as pd
import numpy as np
import os
import joblib
import glob
import nltk
from nltk.corpus import stopwords
import json
import torch
import contractions
import random
from sklearn.model_selection import train_test_split

#nltk.download('stopwords')


ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) 
#ROOT_DIR = 'pan-clef2025/'


TASK_1_DIR = ROOT_DIR + '/dataset/pan25-generative-ai-detection-task1-train/'
TASK_2_DIR = ROOT_DIR + '/dataset/pan25-generative-ai-detection-task2-train/'

OUTPUT_DIR_PATH = ROOT_DIR + '/outputs/'
INPUTS_DIR_PATH = ROOT_DIR + '/inputs/'
DATASETS_DIR_PATH = ROOT_DIR + '/dataset/'


def save_json(data, file_path):
    with open(file_path, "w") as outfile:
        for element in data:  
            json.dump(element, outfile)  
            outfile.write("\n")  

def save_json2(data, file_path):
    def convert_types(obj):
        # Convert NumPy types to native Python types
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    with open(file_path, "w") as outfile:
        for element in data:
            json.dump(element, outfile, default=convert_types)
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

def dataset_doc_len(df):
    df['word_len'] = df['text'].str.split().str.len()
    return df['word_len'].min(), df['word_len'].max(), int(df['word_len'].mean())


# PAN 2025 Dataset Partitions
def build_pan25_dataset():
    #*** Build partitions
    corpus_train_docs = read_json(file_path=TASK_1_DIR + 'original/train.jsonl')
    corpus_train_docs = shuffle(corpus_train_docs)
    corpus_test_docs = read_json(file_path=TASK_1_DIR + 'original/val.jsonl')
    corpus_test_docs = shuffle(corpus_test_docs)
    corpus_train_docs, corpus_val_docs = train_test_split(corpus_train_docs, test_size=0.2, random_state=42, stratify=corpus_train_docs['label'])
    corpus_train_docs.to_csv(TASK_1_DIR + 'train_set.csv')
    corpus_test_docs.to_csv(TASK_1_DIR + 'val_set.csv')
    corpus_val_docs.to_csv(TASK_1_DIR + 'test_set.csv')
    

# Read PAN 2025 Dataset
def read_pan25_dataset(print_info=False):
    train_set = read_csv(TASK_1_DIR + 'train_set.csv')
    val_set = read_csv(TASK_1_DIR + 'val_set.csv')
    test_set = read_csv(TASK_1_DIR + 'test_set.csv')

    train_set = train_set.sample(frac=1).reset_index(drop=True)
    val_set = val_set.sample(frac=1).reset_index(drop=True)

    if print_info:
        print("train_set: ", train_set.info())
        print("val_set: ", val_set.info())
        print("test_set: ", test_set.info())
        print("total_distro_train_val_test: ", train_set.shape, val_set.shape, test_set.shape)
        print("label_distro_train_val_test: ", train_set.value_counts('label'), val_set.value_counts('label'), test_set.value_counts('label'))
        print("source_distro_train_val_test: ", train_set.value_counts('genre'), val_set.value_counts('genre'), test_set.value_counts('genre'))
        print("model_distro_train_val_test: ", train_set.value_counts('model'), val_set.value_counts('model'), test_set.value_counts('model'))
        
        # Model distribution for each genre/source
        print("Model distribution per genre in Train set:\n", train_set.groupby("genre")["model"].value_counts())
        print("Model distribution per genre in Validation set:\n", val_set.groupby("genre")["model"].value_counts())
        print("Model distribution per genre in Test set:\n", test_set.groupby("genre")["model"].value_counts())

        # Label distribution for each source
        print("Label distribution per genre in Train set:\n", train_set.groupby("genre")["label"].value_counts())
        print("Label distribution per genre in Validation set:\n", val_set.groupby("genre")["label"].value_counts())
        print("Label distribution per genre in Test set:\n", test_set.groupby("genre")["label"].value_counts())

    train_set['word_len'] = train_set['text'].str.split().str.len()
    val_set['word_len'] = val_set['text'].str.split().str.len()
    test_set['word_len'] = test_set['text'].str.split().str.len()
    if print_info: 
        print("min_max_avg_token Train: ", train_set['word_len'].min(), train_set['word_len'].max(), int(train_set['word_len'].mean()))
        print("min_max_avg_token Val:   ", val_set['word_len'].min(), val_set['word_len'].max(),  int(val_set['word_len'].mean()))
        print("min_max_avg_token Test:  ", test_set['word_len'].min(), test_set['word_len'].max(), int(test_set['word_len'].mean()))

    return train_set, val_set, test_set


def read_autext_dataset(print_info=False):
    autext_train_set = read_csv(file_path=f'{DATASETS_DIR_PATH}autext2023/subtask1/train_set.csv') 
    autext_val_set = read_csv(file_path=f'{DATASETS_DIR_PATH}autext2023/subtask1/val_set.csv') 
    autext_test_set = read_csv(file_path=f'{DATASETS_DIR_PATH}autext2023/subtask1/test_set.csv') 
    
    autext_train_set = autext_train_set.sample(frac=1).reset_index(drop=True)
    autext_val_set = autext_val_set.sample(frac=1).reset_index(drop=True)
    #autext_test_set = autext_test_set.sample(frac=1).reset_index(drop=True)
    
    autext_train_set.rename(columns={'domain': 'genre'}, inplace=True)
    autext_val_set.rename(columns={'domain': 'genre'}, inplace=True)
    autext_test_set.rename(columns={'domain': 'genre'}, inplace=True)

    if print_info:
        print("autext_train_set: ", autext_train_set.info())
        print("autext_val_set: ", autext_val_set.info())
        print("autext_test_set: ", autext_test_set.info())
        print("total_distro_train_val_test: ", autext_train_set.shape, autext_val_set.shape, autext_test_set.shape)
        print("label_distro_train_val_test: ", autext_train_set.value_counts('label'), autext_val_set.value_counts('label'), autext_test_set.value_counts('label'))
        print("source_distro_train_val_test: ", autext_train_set.value_counts('genre'), autext_val_set.value_counts('genre'), autext_test_set.value_counts('genre'))
        print("model_distro_train_val_test: ", autext_train_set.value_counts('model'), autext_val_set.value_counts('model'), autext_test_set.value_counts('model'))
        
        # Model distribution for each source
        print("Model distribution per genre in Train set:\n", autext_train_set.groupby("genre")["model"].value_counts())
        print("Model distribution per genre in Validation set:\n", autext_val_set.groupby("genre")["model"].value_counts())
        print("Model distribution per genre in Test set:\n", autext_test_set.groupby("genre")["model"].value_counts())

        # Label distribution for each source
        print("Label distribution per genre in Train set:\n", autext_train_set.groupby("genre")["label"].value_counts())
        print("Label distribution per genre in Validation set:\n", autext_val_set.groupby("genre")["label"].value_counts())
        print("Label distribution per genre in Test set:\n", autext_test_set.groupby("genre")["label"].value_counts())

    autext_train_set['word_len'] = autext_train_set['text'].str.split().str.len()
    autext_val_set['word_len'] = autext_val_set['text'].str.split().str.len()
    autext_test_set['word_len'] = autext_test_set['text'].str.split().str.len()
    if print_info: 
        print("min_max_avg_token Train: ", autext_train_set['word_len'].min(), autext_train_set['word_len'].max(), int(autext_train_set['word_len'].mean()))
        print("min_max_avg_token Val:   ", autext_val_set['word_len'].min(), autext_val_set['word_len'].max(),  int(autext_val_set['word_len'].mean()))
        print("min_max_avg_token Test:  ", autext_test_set['word_len'].min(), autext_test_set['word_len'].max(), int(autext_test_set['word_len'].mean()))
    
    return autext_train_set, autext_val_set, autext_test_set

def to_lowercase(text):
    return text.lower()

def handle_contraction_apostraphes(text):
    text = re.sub('([A-Za-z]+)[\'`]([A-Za-z]+)', r'\1'r'\2', text)
    return text

def handle_contraction(text):
  expanded_words = []
  for word in text.split():
    expanded_words.append(contractions.fix(word))
  return ' '.join(expanded_words)

def remove_blank_spaces(text):
    return re.sub(r'\s+', ' ', text).strip() # remove blank spaces

def remove_html_tags(text):
    return re.compile('<.*?>').sub(r'', text) # remove html tags

def remove_special_chars(text):
    text = re.sub('[^A-Za-z0-9]+ ', ' ', text) # remove special chars
    text = re.sub('\W+', ' ', text) # remove special chars
    text = text.replace('"'," ")
    text = text.replace('('," ")
    text = re.sub(r'\s+', ' ', text).strip() # remove blank spaces
    return text

def remove_stop_words(text):
    # remove stop words
    tokens = nltk.word_tokenize(text)
    without_stopwords = [word for word in tokens if not word.lower().strip() in set(stopwords.words('english'))]
    text = " ".join(without_stopwords)
    return text

def text_normalize(text, special_chars=False, stop_words=False):
    text = to_lowercase(text)
    text = handle_contraction(text)
    text = handle_contraction_apostraphes(text)
    text = remove_blank_spaces(text)
    text = remove_html_tags(text)
    if special_chars:
        text = remove_special_chars(text)
    if stop_words: 
        text = remove_stop_words(text)
    return text

def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)