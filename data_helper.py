import numpy
import itertools
import os.path
import sys
from collections import Counter
from constants import *
from datetime import datetime
import sqlite3
from pre_train import pre_train

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

    connector = sqlite3.connect("cancer_train.db")
    connector.text_factory = str
    c = connector.cursor()
    
    select_sql = "select A.ID, group_concat(A.word,' '), B.Class from vocab A, cancer_class B "
    select_sql = select_sql + "where A.ID= B.ID and A.pos not in( ')','(',',','SENT','IN', 'IN/that' ) "
    select_sql = select_sql + "group by A.ID order by Class,A.ID;"

    c.execute(select_sql)
    result = c.fetchall()

    contents = []
    labels = []
    for row in result:
        contents.append(row[1].split())
        labels.append(one_hot_vec(int(row[2])))
        
    connector.close()
    
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
