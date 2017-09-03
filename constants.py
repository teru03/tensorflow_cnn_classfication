NUM_TESTS         = 50
NUM_CLASSES       = 9
#NUM_EPOCHS        = 2
NUM_EPOCHS        = 1000
NUM_MINI_BATCH    = 100
#EMBEDDING_SIZE    = 128
#NUM_FILTERS       = 128
EMBEDDING_SIZE    = 96
NUM_FILTERS       = 96
FILTER_SIZES      = [ 3, 4, 5 ]
L2_LAMBDA         = 0.0001
EVALUATE_EVERY    = 50
CHECKPOINTS_EVERY = 100

SUMMARY_LOG_DIR = 'summary_log'
CHECKPOINTS_DIR = 'checkpoints'

RAW_FILE        = 'data/raw.txt'
DATA_FILE       = 'data/data.npy'
LABEL_FILE      = 'data/labels.npy'
DICTIONARY_FILE = 'data/dictionaries.npy'
TESTDATA_FILE   = 'data/testdata.npy'
