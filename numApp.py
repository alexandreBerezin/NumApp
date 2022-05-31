import FeaturesExtractor.main as fe
from FeatureMatching.distance import getNbAndP,getFilteredMatch

import numpy as np


def getDistance(param:dict,path1:str,path2:str)->float:
    #Calcul des features
    [img1,feat1] = fe.getFeatures(param,path1)
    [img2,feat2] = fe.getFeatures(param,path2)

    #Calcul des match
    kp1,kp2,match = getFilteredMatch(img1,feat1,img2,feat2)

    #Calcul de la distance
    [nb,p] = getNbAndP(kp1,kp2,match)

    return -(nb-p)

def getDistanceMatrix(param:dict,pathList:list)->np.ndarray:
    n = len(pathList)
    D = np.empty((n,n))
    D.fill(np.nan)

    for i in range(n):
        for j in range(i):
            path1 = pathList[i]
            path2 = pathList[j]
            D[i,j]=getDistance(param,path1,path2)

    
    return D
