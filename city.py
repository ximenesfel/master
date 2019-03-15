import os
import sys
import itertools
import math
import logging
import json
import re
import random
import time
import concurrent.futures
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon
import imgaug
from imgaug import augmenters as iaa
import cv2
import skimage
import imutils
from imutils import paths

# Import Mask RCNN
from mrcnn import utils
from mrcnn import visualize
from mrcnn.config import Config
from mrcnn.visualize import display_images
from mrcnn import model as modellib
from mrcnn.model import log

class CityConfig(Config):

	# give the configuration a recognizable name
	NAME = "city"

	# set the number of GPUs to use training along with the number of
	# images per GPU (which may have to be tuned depending on how
	# much memory your GPU has)
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
	NUM_CLASSES = 1 + 2


class CityDataset(utils.Dataset):

	def __init__(self, imagePaths, classNames, width=1024):
		# call the parent constructor
		super().__init__(self)

		# store the image paths and class names along with the width
		# we'll resize images to
		self.imagePaths = imagePaths
		self.classNames = classNames
		self.width = width

		# load the annotation data
		#self.annots = self.load_annotation_data(annotPath)

	# def load_annotation_data(self, annotPath):
	# 	# load the contents of the annotation JSON file (created
	# 	# using the VIA tool) and initialize the annotations
	# 	# dictionary
	# 	annotations = json.loads(open(annotPath).read())
	# 	annots = {}
	#
	# 	# loop over the file ID and annotations themselves (values)
	# 	for (fileID, data) in sorted(annotations.items()):
	# 		# store the data in the dictionary using the filename as
	# 		# the key
	# 		annots[data["filename"]] = data
	#
	# 	# return the annotations dictionary
	# 	return annots

	def load_city(self, idxs):
		"""Load a subset of the Balloon dataset.
		        dataset_dir: Root directory of the dataset.
		        subset: Subset to load: train or val
		        """
		# Add classes. We have only one class to add.
		#self.add_class("city", 1, "people")
		#self.add_class("city", 2, "car")
		# loop over all class names and add each to the 'pills'
		# dataset
		for (classID, label) in self.classNames.items():
			self.add_class("city", classID, label)

		allAnnottations = []

		# Train or validation dataset?
		#dataset_dir = os.path.join(dataset_dir)

		# Load annotations
		# VGG Image Annotator (up to version 1.6) saves each image in the form:
		# { 'filename': '28503151_5b5b7ec140_b.jpg',
		#   'regions': {
		#       '0': {
		#           'region_attributes': {},
		#           'shape_attributes': {
		#               'all_points_x': [...],
		#               'all_points_y': [...],
		#               'name': 'polygon'}},
		#       ... more regions ...
		#   },
		#   'size': 100202
		# }

		for i in idxs:
			allAnnottations.append(annotations[i])

		# # Add images
		for a in allAnnottations:
			# Get the x, y coordinaets of points of the polygons that make up
			# the outline of each object instance. These are stores in the
			# shape_attributes (see json format above)
			# The if condition is needed to support VIA versions 1.x and 2.x.


			if type(a['regions']) is dict:
				polygons = [r['shape_attributes'] for r in a['regions'].values()]
			else:
				polygons = [r['shape_attributes'] for r in a['regions']]

			names = [r['region_attributes'] for r in a['regions']]

			# load_mask() needs the image size to convert polygons to masks.
			# Unfortunately, VIA doesn't include it in JSON, so we must read
			# the image. This is only managable since the dataset is tiny.
			image_path = os.path.join(IMAGES_PATH, a['filename'])
			image = skimage.io.imread(image_path)
			height, width = image.shape[:2]

			self.add_image(
				"city",
				image_id=a['filename'],  # use file name as a unique image id
				path=image_path,
				width=width, height=height,
				polygons=polygons,
				names=names)

	def load_mask(self, image_id):
		"""Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
		# If not a balloon dataset image, delegate to parent class.
		image_info = self.image_info[image_id]
		if image_info["source"] != "city":
			return super(self.__class__, self).load_mask(image_id)

		# Convert polygons to a bitmap mask of shape
		# [height, width, instance_count]
		info = self.image_info[image_id]
		class_names = info["names"]
		mask = np.zeros([info["height"], info["width"], len(info["polygons"])],dtype=np.uint8)

		for i, p in enumerate(info["polygons"]):
			# Get indexes of pixels inside the polygon and set them to 1
			rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
			mask[rr, cc, i] = 1

		# Assign class_ids by reading class_names
		class_ids = np.zeros([len(info["polygons"])])
		# In the surgery dataset, pictures are labeled with name 'a' and 'r' representing arm and ring.
		for i, p in enumerate(class_names):
			# "name" is the attributes name decided when labeling, etc. 'region_attributes': {name:'a'}

			if p['name'] == 'person':
			    class_ids[i] = 1
			elif p['name'] == 'car':
				class_ids[i] = 2
        # assert code here to extend to other labels
		class_ids = class_ids.astype(int)

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
		return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

def load_annotation_data(annotPath):
	# load the contents of the annotation JSON file (created
	# using the VIA tool) and initialize the annotations
	# dictionary
	annotations = json.loads(open(annotPath).read())
	annots = {}

	# loop over the file ID and annotations themselves (values)
	for (fileID, data) in sorted(annotations.items()):
		# store the data in the dictionary using the filename as
		# the key
		annots[data["filename"]] = data

	# return the annotations dictionary
	return annots

# initialize the dataset path, images path, and annotations file path
DATASET_PATH = os.path.abspath("dataset")
IMAGES_PATH = os.path.sep.join([DATASET_PATH, "image"])
ANNOT_PATH = os.path.sep.join([DATASET_PATH, "via_region_data.json"])

# initialize the class names dictionary
CLASS_NAMES = {1: "person", 2: "car"}

# initialize the path to the Mask R-CNN pre-trained on COCO
COCO_PATH = "mask_rcnn_coco.h5"

# initialize the name of the directory where logs and output model
# snapshots will be stored
LOGS_AND_MODEL_DIR = "output"

# initialize the amount of data to use for training
TRAINING_SPLIT = 0.75

# grab all image paths, then randomly select indexes for both training
# and validation
IMAGE_PATHS = sorted(list(paths.list_images(IMAGES_PATH)))

annotations = load_annotation_data(ANNOT_PATH)
annotations = list(annotations.values())  # don't need the dict keys

annotations = [a for a in annotations if a['regions']]

idxs = list(range(0, len(annotations)))
random.seed(42)
random.shuffle(idxs)
i = int(len(idxs) * TRAINING_SPLIT)
trainIdxs = idxs[:i]
valIdxs = idxs[i:]


# load the training dataset
trainDataset = CityDataset(IMAGE_PATHS, CLASS_NAMES)
trainDataset.load_city(trainIdxs)
trainDataset.prepare()

# load the validation dataset
valDataset = CityDataset(IMAGE_PATHS, CLASS_NAMES)
valDataset.load_city(valIdxs)
valDataset.prepare()

# initialize the training configuration
config = CityConfig()
config.display()

# initialize the model and load the COCO weights so we can
# perform fine-tuning
model = modellib.MaskRCNN(mode="training", config=config,model_dir=LOGS_AND_MODEL_DIR)
model.load_weights(COCO_PATH, by_name=True,
			exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
				"mrcnn_bbox", "mrcnn_mask"])

# train *just* the layer heads
model.train(trainDataset, valDataset, epochs=10, layers="heads", learning_rate=config.LEARNING_RATE)

# unfreeze the body of the network and train *all* layers
model.train(trainDataset, valDataset, epochs=20, layers="all", learning_rate=config.LEARNING_RATE / 10)