# USAGE
# python build_googlenet_covid19.py

# import the necessary packages
from config import googlenet_covid19_config as config
from sklearn.preprocessing import LabelEncoder
from pyimagesearch.preprocessing import AspectAwarePreprocessor
from sklearn.model_selection import train_test_split
from pyimagesearch.io import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import json
#import codecs
import cv2
import os
#import nujson as ujson

# grab the paths to the training images, then extract the training
# class labels and encode them
trainPaths = list(paths.list_images(config.TRAIN_IMAGES))
trainLabels = [p.split(os.path.sep)[-2] for p in trainPaths]
valPaths = list(paths.list_images(config.VAL_IMAGES))
valLabels = [p.split(os.path.sep)[-2] for p in valPaths]
le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)
'''
for valPath in valPaths:
	# extract the class label from the filename
	label = valPath.split(os.path.sep)[-2]

	# load the image, swap color channels, and resize it to be a fixed
	# 224x224 pixels while ignoring aspect ratio
	image = cv2.imread(valPath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (64, 64))
    
	# update the data and labels lists, respectively
	data.append(image)
	labels.append(label)
 '''   
# perform stratified sampling from the training set to construct a
# a testing set
split = train_test_split(trainPaths, trainLabels,
	test_size=config.NUM_TEST_IMAGES, stratify=trainLabels,
	random_state=42)
(trainPaths, testPaths, trainLabels, testLabels) = split
'''
split = train_test_split(trainPaths, trainLabels,
	test_size=config.NUM_VAL_IMAGES, stratify=trainLabels,
	random_state=42)
(trainPaths, valPaths, trainLabels, valLabels) = split
'''
# load the validation filename => class from file and then use these
# mappings to build the validation paths and label 
# load the validation filename => class from file and then use these
# mappings to build the validation paths and label lists
'''
M = open(config.VAL_MAPPINGS).read().strip().split("\n")
M = [r.split("\t")[:2] for r in M]
valPaths = [os.path.sep.join([config.VAL_IMAGES, m[0]]) for m in M]
valLabels = le.transform([m[1] for m in M])
'''
# construct a list pairing the training, validation, and testing
# image paths along with their corresponding labels and output HDF5
# files
datasets = [
	("train", trainPaths, trainLabels, config.TRAIN_HDF5),
	("val", valPaths, valLabels, config.VAL_HDF5),
	("test", testPaths, testLabels, config.TEST_HDF5)]


# initialize the lists of RGB channel averages
aap = AspectAwarePreprocessor(64, 64)
(R, G, B) = ([], [], [])

# loop over the dataset tuples
for (dType, paths, labels, outputPath) in datasets:
	# create HDF5 writer
    print("[INFO] building {}...".format(outputPath))
    writer = HDF5DatasetWriter((len(paths), 64, 64, 3), outputPath)

	# initialize the progress bar
    widgets = ["Building Dataset: ", progressbar.Percentage(), " ",
        progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(paths),
		widgets=widgets).start()
    # loop over the image paths
    for (i, (path, label)) in enumerate(zip(paths, labels)):
		# load the image from disk
        image = cv2.imread(path)
        image = aap.preprocess(image)

		# if we are building the training dataset, then compute the
		# mean of each channel in the image, then update the
		# respective lists
        if dType == "train":
            (b, g, r) = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)

		# add the image and label to the HDF5 dataset
        writer.add([image], [label])
        pbar.update(i)

	# close the HDF5 writer
    pbar.finish()
    writer.close()

# construct a dictionary of averages, then serialize the means to a
# JSON file
print("[INFO] serializing means...")
'''
R_J = np.mean(R, dtype=np.float64)
G_J = np.mean(G, dtype=np.float64)
B_J = np.mean(B, dtype=np.float64)
'''
D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
f = open(config.DATASET_MEAN, "w")
f.write(json.dumps(D))
f.close()