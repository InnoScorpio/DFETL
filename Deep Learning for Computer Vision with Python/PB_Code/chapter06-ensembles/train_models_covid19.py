# USAGE
# python train_models_mod.py --dataset ../datasets/covid19-dataset/training --output output --models models

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import AspectAwarePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from pyimagesearch.nn.conv import MiniVGGNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from imutils import paths
#from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
	help="path to output directory")
ap.add_argument("-m", "--models", required=True,
	help="path to output models directory")
ap.add_argument("-n", "--num-models", type=int, default=5,
	help="# of models to train")
ap.add_argument("-d", "--dataset", required=True, 
    help="path to output directory")
args = vars(ap.parse_args())

print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))


# initialize the image preprocessors
aap = AspectAwarePreprocessor(64, 64)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities
# to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0

# load the training and testing data, then scale it into the
# range [0, 1]
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, random_state=25)
'''
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0
'''
# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# initialize the label names for the CIFAR-10 dataset
labelNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
labelNames = [str(x) for x in np.unique(labelNames)]

# convert the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=10, width_shift_range=0.1,
	height_shift_range=0.1, horizontal_flip=True,
	fill_mode="nearest")

# loop over the number of models to train
for i in np.arange(0, args["num_models"]):
	# initialize the optimizer and model
	print("[INFO] training model {}/{}".format(i + 1,
		args["num_models"]))
	opt = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9,
		nesterov=True)
	model = MiniVGGNet.build(width=64, height=64, depth=3,
		classes=len(labelNames))
	model.compile(loss="sparse_categorical_crossentropy", optimizer=opt,
		metrics=["accuracy"])

	# train the network
	H = model.fit_generator(aug.flow(trainX, trainY, batch_size=64),
		validation_data=(testX, testY), epochs=100,
		steps_per_epoch=len(trainX) // 64, verbose=1)

	# save the model to disk
	p = [args["models"], "model_{}.model".format(i)]
	model.save(os.path.sep.join(p))

	# evaluate the network
	predictions = model.predict(testX, batch_size=64)
	report = classification_report(testY.argmax(axis=1),
		predictions.argmax(axis=1), target_names=lb.classes_)

	# save the classification report to file
	p = [args["output"], "model_{}.txt".format(i)]
	f = open(os.path.sep.join(p), "w")
	f.write(report)
	f.close()

	# plot the training loss and accuracy
	p = [args["output"], "model_{}.png".format(i)]
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(np.arange(0, 100), H.history["loss"],
		label="train_loss")
	plt.plot(np.arange(0, 100), H.history["val_loss"],
		label="val_loss")
	plt.plot(np.arange(0, 100), H.history["acc"],
		label="train_acc")
	plt.plot(np.arange(0, 100), H.history["val_acc"],
		label="val_acc")
	plt.title("Training Loss and Accuracy for model {}".format(i))
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend()
	plt.savefig(os.path.sep.join(p))
	plt.close()