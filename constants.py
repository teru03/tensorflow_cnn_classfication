NUM_TESTS         = 50
NUM_CLASSES       = 9
NUM_EPOCHS        = 10
NUM_MINI_BATCH    = 100
EMBEDDING_SIZE    = 64
NUM_FILTERS       = 64
FILTER_SIZES      = [ 2, 3, 4 ]
#FILTER_SIZES      = [ 3, 4 ]
L2_LAMBDA         = 0.0001
EVALUATE_EVERY    = 50
CHECKPOINTS_EVERY = 100

SUMMARY_LOG_DIR = 'summary_log'
CHECKPOINTS_DIR = 'checkpoints'

RAW_FILE        = 'data/raw.txt'
DATA_FILE       = 'data/data.npy'
LABEL_FILE      = 'data/labels.npy'
DICTIONARY_FILE = 'data/dictionaries.npy'
