import numpy as np
import argparse
import cv2
import os
# import colo from "models/colorization_deploy_v1.prototxt";

DIR = r"/Users/sneha_rajaraman/Desktop/colorize"
PROTOTXT = os.path.join(DIR, r"models/colorization_deploy_v1.prototxt")
POINTS = os.path.join(DIR, r"models/pts_in_hull.npy")
MODEL = os.path.join(DIR, r"models/colorization_release_v2.caffemodel")

# Argparser: Provides the path for the input image
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", type=str, required=True,
                    help="input path for black & white image") # allows to pass in different images by providing their path
args = vars(parser.parse_args())

# Loading the Model
print("Loading Model...")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL) # DNN - Deep Neural Network, module in cv2
pts = np.load(POINTS) # loads the numpy object into memory

# Loads centers for ab channel
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1) # creates 1 x 1 convolutions
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype = "float32")]

# Reading the input image
img = cv2.imread(args["image"]) # converts image into a matrix
scaled = img.astype("float32") / 255.0 # scales to a range of values between 0 to 1
lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB) # convert the colors from RGB (BGR in CV2 format) to LAB

# Resizing the images
resize = cv2.resize(lab, (224, 224)) # because the original model is trained on this size of images
L_channel = cv2.split(resize)[0] # getting the L channel (first channel) of the image
L_channel -= 50 # hyperparameter

# Colorizing the image!
print("Colorizing the image!")
net.setInput(cv2.dnn.blobFromImage(L_channel)) # uses a collection of similar images that are differently pre-processed (blobbed from these images and feed image)
ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0)) # feed forward input

# Resize back to original image
ab_channel = cv2.resize(ab_channel, (img.shape[1], img.shape[0]))

L_channel = cv2.split(lab)[0]
colorized = np.concatenate((L_channel[:, :, np.newaxis], ab_channel), axis=2) # append L to ab

colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR) # coonvert from LAB colorspace to BGR colorspace
# cv2 reads in BGR and not in RGB
colorized = np.clip(colorized, 0, 1)