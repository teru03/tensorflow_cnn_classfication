from pre_train import pre_train


traindb = pre_train('train',tagdir="../tree-tagger")
traindb.create_db()
traindb.tagger()

testdb = pre_train('test',tagdir="../tree-tagger")
testdb.create_db()
testdb.tagger()

