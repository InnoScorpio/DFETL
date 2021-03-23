# import the necessary packages
from os import path

# define the paths to the training and validation directories
TRAIN_IMAGES = "../datasets/covid19-dataset/testing"
VAL_IMAGES = "../datasets/covid19-dataset/training"
'''
# define the path to the file that maps validation filenames to
# their corresponding class labels
VAL_MAPPINGS = "../datasets/tiny-imagenet-200/val/val_annotations.txt"

# define the paths to the WordNet hierarchy files which are used
# to generate our class labels
WORDNET_IDS = "../datasets/tiny-imagenet-200/wnids.txt"
WORD_LABELS = "../datasets/tiny-imagenet-200/words.txt"
'''
# since we do not have access to the testing data we need to
# take a number of images from the training data and use it instead
NUM_CLASSES = 2
NUM_TEST_IMAGES = 15 * NUM_CLASSES
NUM_VAL_IMAGES = 15 * NUM_CLASSES

# define the path to the output training, validation, and testing
# HDF5 files
TRAIN_HDF5 = "../datasets/covid19-dataset/hdf5/resnet_train.hdf5"
VAL_HDF5 = "../datasets/covid19-dataset/hdf5/resnet_val.hdf5"
TEST_HDF5 = "../datasets/covid19-dataset/hdf5/resnet_test.hdf5"

# define the path to the dataset mean
DATASET_MEAN = "output/resnet-covid19-200-mean.json"

# define the path to the output directory used for storing plots,
# classification reports, etc.
OUTPUT_PATH = "output"
MODEL_PATH = path.sep.join([OUTPUT_PATH,
	"checkpoints/epoch_20.hdf5"])
FIG_PATH = path.sep.join([OUTPUT_PATH,
	"resnet56_covid19.png"])
JSON_PATH = path.sep.join([OUTPUT_PATH,
	"resnet56_covid19.json"])