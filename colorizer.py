import numpy as np
import argparse
import cv2
import os
# import PROTOTXT from "models/colorization_deploy_v1.prototxt";

DIR = r"/Users/sneha_rajaraman/Desktop/colorize"
PROTOTXT = os.path.join(DIR, r"models/colorization_deploy_v1.prototxt")
POINTS = os.path.join(DIR, r"models/pts_in_hull.npy")
MODEL = os.path.join(DIR, r"models/colorization_release_v2.caffemodel")