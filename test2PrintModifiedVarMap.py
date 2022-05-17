'''
affiche
'''

from cv2 import CONTOURS_MATCH_I2
import numpy as np
from FeaturesExtractor import Kernel as k
from FeaturesExtractor import Features as f
from ImageProcessing import processing as pr
from misc import *

import matplotlib.pyplot as plt


contours = pr.preprocess("data/imageBase.png",50,3)
shape = np.shape(contours)

Xvec = XVecFromArray(contours)

weightVec = contours.ravel()

#Creation Ker
ker = k.Kernel(weightVec,Xvec,3)

varVect = ker.getVarVector()


##Creation ext
ext = f.Extractor(varVect,ker)

idxF = ext.getIndexFirstFeature()

ext.addFeature(idxF)



newVect = ext.getVarVector()

varmap = np.reshape(varVect,shape)

newVarMap = np.reshape(newVect,shape)

plt.subplot(1,2,1)
plt.imshow(varmap)
x,y = Xvec[idxF]
plt.scatter(y,x,color='r')

plt.subplot(1,2,2)
plt.imshow(newVarMap)

plt.show()

