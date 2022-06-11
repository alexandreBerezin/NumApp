'''
Get 10 features and record them
'''

from cv2 import CONTOURS_MATCH_I2
import numpy as np
from FeaturesExtractor import Kernel as k
from FeaturesExtractor import Features as f
from ImageProcessing import processing as pr
from FeaturesExtractor.misc import *

import matplotlib.pyplot as plt

### paramètres ###
#coté de l'image en pixels
cote = 100
#flou gaussien
flou_sigma = 3
#seul pour le calcul de covariance
treshold = 10
# longeur caractéristique pour le kernel
l = 3




contours = pr.preprocess("data/imageBase.png",200,3)
shape = np.shape(contours)

Xvec = XVecFromArray(contours)

weightVec = contours.ravel()

treshold = 10

#Creation Ker
ker = k.Kernel(weightVec,Xvec,3,treshold)

varVect = ker.getVarVector()

ext = f.Extractor(varVect,ker)

l = ext.getFeature(3)
print("liste finale:")
print(l)

