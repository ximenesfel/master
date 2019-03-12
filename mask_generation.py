import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import progressbar


import matplotlib.image as mpimg
from PIL import Image

from config.config import Config


# Load the configuration
config = Config()


# Read all the files in segmentation folder
#segmentationImagePath = [os.path.abspath(x) for x in os.listdir(config.datasetSegmentationPath)]
print("[INFO] Reading files in segmentation folder ...")
segmentationImagePath = os.listdir(config.datasetSegmentationPath)

# Create a folder called Mask
os.mkdir("./Mask")
os.chdir("./Mask")

# Create a new folder for label and enter inside the folder
os.mkdir("{}".format(config.label))
os.chdir("{}".format(config.label))

# Initialize the progress bar
widgets = ["Building mask for {}".format(config.label), progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(segmentationImagePath), widgets=widgets).start()

i = 0

for imageNumber in segmentationImagePath:

    # Create a folder
    os.mkdir("{}".format(format(str(imageNumber.replace("segmentation","").replace(".png","")))))
    os.chdir("{}".format(format(str(imageNumber.replace("segmentation", "").replace(".png", "")))))

    # Read the image
    img = cv2.imread(config.datasetSegmentationPath + imageNumber, cv2.IMREAD_COLOR)

    # copy the image
    imgCopy = img.copy()

    # Index value
    indexValue = 0


    # Walkthrough all pixel values in list
    for pixelValue in config.pixelsValuesToSegment:

        maskData = np.where(imgCopy == [pixelValue[0], pixelValue[1], pixelValue[2]], 1.,  0.)
        #cropMaskData = maskData[:,:,0]

        obtainedMask = bool(maskData.any())

        if ( obtainedMask ):
            plt.imsave("mask_{}.png".format(str(imageNumber.replace("segmentation","").replace(".png","")) + "_" + str(indexValue)), maskData)

        indexValue += 1

    pbar.update(i)
    i += 1

    os.chdir("..")


pbar.finish()
