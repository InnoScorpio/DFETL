# define the paths to the images directory
IMAGES_PATH = "../datasets/covid19-dataset/"

# since we do not have validation data or access to the testing
# labels we need to take a number of images from the training
# data and use them instead
NUM_CLASSES = 2
NUM_VAL_IMAGES = 15 * NUM_CLASSES
NUM_TEST_IMAGES = 15 * NUM_CLASSES

# define the path to the output training, validation, and testing
# HDF5 files
TRAIN_HDF5 = "../datasets/covid19-dataset/hdf5/train.hdf5"
VAL_HDF5 = "../datasets/covid19-dataset/hdf5/val.hdf5"
TEST_HDF5 = "../datasets/covid19-dataset/hdf5/test.hdf5"

# path to the output model file
MODEL_PATH = "output/alexnet_covid19-dataset.model"

# define the path to the dataset mean
DATASET_MEAN = "output/covid19-dataset.json"

# define the path to the output directory used for storing plots,
# classification reports, etc.
OUTPUT_PATH = "output"