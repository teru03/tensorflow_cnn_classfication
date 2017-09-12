# coding: UTF-8

import pandas as pd
import numpy as np
import scipy
import csv
import json
import sqlite3
import treetaggerwrapper

class pre_train:

    def __init__(self,mode,tagdir,input_dir='./input'):
        
        self.mode = mode
        
        if mode == 'train':
            self.db_name = "cancer_train.db"
            self.text_file = input_dir+"/training_text"
            self.class_file = input_dir+"/training_variants.csv"
        else:
            self.db_name = "cancer_test.db"
            self.text_file = input_dir+"/test_text"
        
        self.tagdir = tagdir
        self.input_dir = input_dir

    def create_db(self):
        
        connector = sqlite3.connect(self.db_name)
        connector.text_factory = str
        c = connector.cursor()
        sql = 'create table if not exists cancer_text (ID int, Text blob);'
        c.execute(sql)

        tdata = pd.read_csv(self.text_file, sep='\|\|', engine='python', 
                          skiprows=1, names=['ID', 'Text'])

        print tdata.head()

        sql = 'insert into cancer_text values(?,?) '
                    
        for i in xrange(len(tdata.index)):
            val = (int(tdata['ID'][i]),tdata['Text'][i])
            c.execute(sql,val)

        connector.commit()

        if self.mode == 'train':
            cdata = pd.read_csv(self.class_file, sep=',', skiprows=1, names=['ID','Gene','Variation','Class'] )
            print cdata.head()

            sql = 'create table if not exists cancer_class (ID int, Class varchar);'
            c.execute(sql)

            sql = ' insert into cancer_class values(?,?) '
            for i in xrange(len(cdata['Class'].index)):
                val = (int(cdata['ID'][i]),cdata['Class'][i])
                c.execute(sql,val)

            connector.commit()

        connector.close()

    def chunked(self, text, n):
        return [ text[x:x+n] for x in range(0,len(text),n)]

    def tagger(self):

        #TAGDIRにはTreeTaggerをインストールしたディレクトリを指定。
        tagger = treetaggerwrapper.TreeTagger(TAGLANG='en',TAGDIR=self.tagdir)

        connector = sqlite3.connect(self.db_name)
        connector.text_factory = str
        c = connector.cursor()
        
        create_sql = "create table if not exists vocab( ID int, word varchar, pos varchar, res varchar );"
        c.execute(create_sql)

        select_sql = "select ID, Text from cancer_text where ID > (select max(ID) from vocab); "
        c.execute(select_sql)
        result = c.fetchall()

        if( len(result) == 0 ):
           select_sql = "select ID, Text from cancer_text"
           c.execute(select_sql)
           result = c.fetchall()
            
        insert_sql = "insert into vocab values(?,?,?,?);"

        for row in result:
            ID = row[0]

            if( len(row[1]) >= 300000 ):
                datas=self.chunked(row[1], 200000)
            else:
                datas=[row[1]]

            for data in datas:
               tags = tagger.TagText(data)#解析したいテキストを引数に
               for tag in tags:
                   tagary = tag.split('\t')
                   if( len(tagary) == 3 ):
                       val=(ID,tagary[0],tagary[1],tagary[2])
                       c.execute(insert_sql,val)

            connector.commit()

        connector.close()    

    def get_db_name(self):
        return self.db_name
    
