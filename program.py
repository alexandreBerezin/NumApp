import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt

import os
import sys

#Ajout du chemin 
NumAppPath = os.path.abspath('./NumApp')
sys.path.append(NumAppPath)

from NumApp.numApp import getDistanceMatrix






param = {
    ##### Algo Param
    "longeur RBF":6,
    "nombre features":200,
    "denoiseTV weight": 0.2,
    "alpha": 7,
    "reprojThreshold":4,
    
    
    ##### Config param
    "Features Folder":"featuresTest2/",
    "Data Folder":"dataTest2",
    "Multiprocess Pool":12,
        
}

dataPath = os.path.join(NumAppPath,param["Data Folder"])


#D = getDistanceMatrix(param)

#np.save("DistMatrixTest2",D)