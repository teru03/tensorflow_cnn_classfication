# coding:utf-8
import numpy
import itertools
import os.path
import sys
from collections import Counter
from constants import *
from datetime import datetime
from data_helper import log
import tensorflow as tf
import sqlite3
import data_helper
import csv



def split_word(tagger, content):
    word = tagger.parse(content).split(' ')
    word = [ w.strip().decode('utf-8') for w in word ]
    return word

def one_hot_vec(index):
    v = numpy.zeros(NUM_CLASSES)
    v[index] = 1
    return v

def padding(contents, max_word_count):
    padded_contents = []
    for i in xrange(len(contents)):
        content = contents[i]
        padded_contents.append(content + [ '<PAD/>' ] * (max_word_count - len(content)))

    return padded_contents

if __name__ == '__main__':

    print('--start')

    dmyx, dmyy, d = data_helper.load_data_and_labels_and_dictionaries()
    if os.path.exists(TESTDATA_FILE) :
        print "load data"
        testdata    = numpy.load(TESTDATA_FILE)
        x_dim = len(testdata[0])
        print "x_dim ",x_dim

    else :

        dictionaries_inv = { c: i for i, c in enumerate(d) }
    
        connector = sqlite3.connect("./cancer_test.db")
        connector.text_factory = str
        c = connector.cursor()
    
        select_sql = "select ID, group_concat(word,' ') from vocab "
        select_sql = select_sql + " where pos not in( ')', '(', ',', 'SENT','IN', 'IN/that' )"
        select_sql = select_sql + " group by ID order by ID;"

        c.execute(select_sql)
        result = c.fetchall()

        contents = []
        for row in result:
            content = row[1].split()
            contents.append(content)

        connector.close()
        
        x_dim =  max([ len(c) for c in contents ])
        print 'x_dim',x_dim
        contents = padding(contents, max([ len(c) for c in contents ]))

        x_test = []
        for content in contents:
            inv_d = []
            for word in content:
                if dictionaries_inv.has_key(word):
                    inv_d.append(dictionaries_inv[word])    
                else:
                    inv_d.append(dictionaries_inv['<PAD/>'])
            x_test.append(inv_d)
        testdata = numpy.array(x_test)
        numpy.save(TESTDATA_FILE,testdata)

    keep = tf.placeholder(tf.float32)
    input_x = tf.placeholder(tf.int32,   [ None, x_dim ])

    with tf.name_scope('embedding'):
        w  = tf.Variable(tf.random_uniform([ len(d), EMBEDDING_SIZE ], -1.0, 1.0), name='weight')
        e  = tf.nn.embedding_lookup(w, input_x)
        ex = tf.expand_dims(e, -1)

    p_array = []
    for filter_size in FILTER_SIZES:
        with tf.name_scope('conv-%d' % filter_size):
            w  = tf.Variable(tf.truncated_normal([ filter_size, EMBEDDING_SIZE, 1, NUM_FILTERS ], stddev=0.02), name='weight')
            b  = tf.Variable(tf.constant(0.1, shape=[ NUM_FILTERS ]), name='bias')
            c0 = tf.nn.conv2d(ex, w, [ 1, 1, 1, 1 ], 'VALID')
            c1 = tf.nn.relu(tf.nn.bias_add(c0, b))
            c2 = tf.nn.max_pool(c1, [ 1, x_dim - filter_size + 1, 1, 1 ], [ 1, 1, 1, 1 ], 'VALID')
            p_array.append(c2)

    p = tf.concat(p_array,3)

    with tf.name_scope('fc'):
        total_filters = NUM_FILTERS * len(FILTER_SIZES)
        w = tf.Variable(tf.truncated_normal([ total_filters, NUM_CLASSES ], stddev=0.02), name='weight')
        b = tf.Variable(tf.constant(0.1, shape=[ NUM_CLASSES ]), name='bias')
        h0 = tf.nn.dropout(tf.reshape(p, [ -1, total_filters ]), keep)
        predict_y = tf.nn.softmax(tf.matmul(h0, w) + b)

    global_step = tf.Variable(0, name="global_step", trainable=False)

    sess = tf.InteractiveSession()

    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer)
    saver.restore(sess, CHECKPOINTS_DIR + '/model-last')

    datalen = len(testdata)
    divn = 50
    print "datalen = %d divn + %d"%(datalen,divn)
    allpred = []
    for i in xrange(int(datalen/divn)):
        print "pred no %d - %d"%(i*divn,(i+1)*divn)

        xdata = testdata[i*divn:(i+1)*divn]
        #print "len = %d"%len(xdata)
        #print xdata
        predictions = sess.run(
            predict_y,
            feed_dict={ input_x: xdata, keep:1.0}
        )
        allpred.extend(predictions)

    remain = datalen%divn
    print "pred remain %d %d"%(remain,(datalen-remain))

    xdata = testdata[datalen-remain:]
    predictions = sess.run(
        predict_y,
        feed_dict={ input_x: xdata, keep:1.0}
    )
    allpred.extend(predictions)

    with open('submit.csv', 'wt') as outf:
        fo = csv.writer(outf, lineterminator='\n')
        title = ['ID','class1','class2','class3','class4','class5','class6','class7','class8','class9']
        fo.writerow(title)
        for i,pred in enumerate(predictions):
            buff=[i]
            buff.extend(['%.4f'%prob for prob in pred ])
            fo.writerow(buff)

    print '-- end'
    

