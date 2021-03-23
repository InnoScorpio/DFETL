# USAGE
# python test_ensemble_covid19.py --models models --dataset ../datasets/covid19-dataset/training

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.models import load_model
from sklearn.model_selection import train_test_split
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import AspectAwarePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from imutils import paths
#from pyimagesearch.datasets import SimpleDatasetLoader
#from keras.datasets import cifar10
import numpy as np
import argparse
import glob
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, 
    help="path to output directory")
ap.add_argument("-m", "--models", required=True,
	help="path to models directory")
args = vars(ap.parse_args())

print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

aap = AspectAwarePreprocessor(64, 64)
iap = ImageToArrayPreprocessor()

sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0

# load the testing data, then scale it into the range [0, 1]
#(testX, testY) = cifar10.load_data()[1]
#testX = testX.astype("float") / 255.0

(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, random_state=42)
'''
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0
'''

# initialize the label names for the CIFAR-10 dataset
#labelNames = ["airplane", "automobile", "bird", "cat", "deer",
#	"dog", "frog", "horse", "ship", "truck"]

classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]

# convert the labels from integers to vectors
lb = LabelBinarizer()
testY = lb.fit_transform(testY)

# construct the path used to collect the models then initialize the
# models list
modelPaths = os.path.sep.join([args["models"], "*.model"])
modelPaths = list(glob.glob(modelPaths))
models = []

# loop over the model paths, loading the model, and adding it to
# the list of models
for (i, modelPath) in enumerate(modelPaths):
	print("[INFO] loading model {}/{}".format(i + 1,
		len(modelPaths)))
	models.append(load_model(modelPath))

# initialize the list of predictions
print("[INFO] evaluating ensemble...")
predictions = []

# loop over the models
for model in models:
	# use the current model to make predictions on the testing data,
	# then store these predictions in the aggregate predictions list
	predictions.append(model.predict(testX, batch_size=64))

# average the probabilities across all model predictions, then show
# a classification report
predictions = np.average(predictions, axis=0)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=classNames))