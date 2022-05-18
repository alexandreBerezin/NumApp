'''
Main
'''

from cv2 import CONTOURS_MATCH_I2
import numpy as np
from FeaturesExtractor import Kernel as k
from FeaturesExtractor import Features as f
from ImageProcessing import processing as pr
from FeaturesExtractor.misc import *

import matplotlib.pyplot as plt


contours = pr.preprocess("data/imageBase.png",10,3)
shape = np.shape(contours)


Xvec = XVecFromArray(contours)

weightVec = contours.ravel()

#Creation Ker
ker = k.Kernel(weightVec,Xvec,3,10)

varVect = ker.getVarVector()


##Creation ext
ext = f.Extractor(varVect,ker)

featureList = ext.getFeature(1)

print(featureList)

'''
idx= [1380, 1432, 1277, 1023, 1480]
X =[]
Y = []
for id in idx:
    x,y = Xvec[id]
    X.append(x)
    Y.append(y)
    
plt.imshow(contours)
plt.scatter(Y,X,color='r')
plt.show()


'''