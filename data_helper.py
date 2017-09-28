import numpy
import itertools
import os.path
import sys
from collections import Counter
from constants import *
from datetime import datetime
import sqlite3
import nltk
from nltk.corpus import stopwords
import pandas as pd
import re


def cleanup(text):
    stop_words = set(stopwords.words('english'))
    letters_only = re.sub("[^a-zA-Z]", " ", text)
    words = letters_only.lower().split()
    text = [ word for word in words if not word in stop_words]
    #words = text.lower().split()
    #text = [ word.lower() for word in words if not "0xe2"]
    return text

def one_hot_vec(index):
    v = numpy.zeros(NUM_CLASSES)
    v[index-1] = 1
    return v

def padding(contents, max_word_count):
    padded_contents = []
    for i in xrange(len(contents)):
        content = contents[i]
        padded_contents.append(content + [ '<PAD/>' ] * (max_word_count - len(content)))

    return padded_contents

def load_data_and_labels_and_dictionaries():
    
    if os.path.exists(DATA_FILE) and os.path.exists(LABEL_FILE) and os.path.exists(DICTIONARY_FILE):
        data         = numpy.load(DATA_FILE)
        labels       = numpy.load(LABEL_FILE)
        dictionaries = numpy.load(DICTIONARY_FILE)
        print "----- data "
        print data
        print "----- labels "
        print labels
        return [ data, labels, dictionaries ]

    nltk.download("stopwords")

    train_v_df = pd.read_csv('./input/training_variants.csv')
    train_t_df = pd.read_csv('./input/training_text',sep='\|\|',
                    skiprows=1,engine='python',names=["ID","text"])
#    test_t_df = pd.read_csv('./input/test_text',sep='\|\|',
#                    skiprows=1,engine='python',names=["ID","text"])
    
    contents = []
    labels = []
    for i in xrange(len(train_t_df['ID'].values)):
        contents.append(cleanup(train_t_df['text'][i]))
        labels.append(one_hot_vec(train_v_df['Class'][i]))
                      
#    for i in xrange(len(test_t_df['ID'].values)):
#        contents.append(cleanup(test_t_df['text'][i]))
                      
    
    print max([ len(c) for c in contents ])
    contents = padding(contents, max([ len(c) for c in contents ]))

    ctr = Counter(itertools.chain(*contents))
    print ctr.most_common()[:10]

    dictionaries     = [ c[0] for c in ctr.most_common() ]
    dictionaries_inv = { c: i for i, c in enumerate(dictionaries) }

    print dictionaries[:10]

    data = [ [ dictionaries_inv[word] for word in content ] for content in contents ]

    data         = numpy.array(data)
    labels       = numpy.array(labels)
    dictionaries = numpy.array(dictionaries)

    numpy.save(DATA_FILE,       data)
    numpy.save(LABEL_FILE,      labels)
    numpy.save(DICTIONARY_FILE, dictionaries)

    return [ data, labels, dictionaries ]

def log(content):
    time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    print time + ': ' + content
    sys.stdout.flush()
