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