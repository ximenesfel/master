

class Config:

    def __init__(self):

        self.datasetSegmentationPath = "/home/ximenes/code/dataset_generation/dataset/Segmentation/"
        self.label = "car"
        #self.pixelsValuesToSegment = [(98,5,104)] # people
        self.pixelsValuesToSegment = [(6,108,153), (130,66,148), (227,146,71)] # car
