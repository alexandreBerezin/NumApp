'''
Test Affichage VarMap
OK 16/05/2022
'''

from cv2 import CONTOURS_MATCH_I2
import numpy as np
from FeaturesExtractor import Kernel as k
from ImageProcessing import processing as pr
from misc import *
import matplotlib.pyplot as plt


contours = pr.preprocess("data/imageBase.png",200)
shape = np.shape(contours)

Xvec = XVecFromArray(contours)

weightVec = contours.ravel()

#Création Ker
ker = k.Kernel(weightVec,Xvec,3.0)

varVect = ker.varVector()
varMap = np.reshape(varVect,shape)
plt.imshow(varMap)
plt.colorbar()
plt.show()
