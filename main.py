from cv2 import CONTOURS_MATCH_I2
import numpy as np
from FeaturesExtractor import Kernel as k
from ImageProcessing import processing as pr
import matplotlib.pyplot as plt


contours = pr.preprocess("data/imageBase.png")
plt.imshow(contours)
plt.show()