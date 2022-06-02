import FeaturesExtractor.main as fe
from FeatureMatching.distance import getNbAndP,getFilteredMatch

import numpy as np


def getDistance(param:dict,path1:str,path2:str)->float:

    alpha = param["alpha"]
    reprojThreshold = param["reprojThreshold"]

    #Calcul des features
    [img1,feat1] = fe.getFeatures(param,path1)
    [img2,feat2] = fe.getFeatures(param,path2)

    #Calcul des match
    kp1,kp2,match = getFilteredMatch(img1,feat1,img2,feat2,reprojThreshold)

    #Calcul de la distance
    [nb,p] = getNbAndP(kp1,kp2,match)

    return -(nb-alpha*np.log(p))

def getDistanceMatrix(param:dict,pathList:list)->np.ndarray:
    n = len(pathList)
    D = np.empty((n,n))
    D.fill(np.nan)


    count = 1

    for i in range(n):
        for j in range(i):
            print(count,"/",(n-1)*n/2)
            path1 = pathList[i]
            path2 = pathList[j]
            D[i,j]=getDistance(param,path1,path2)

            count = count+1

        

    
    return D


def getContraste(D:np.ndarray,sameCoin:np.ndarray)->float:
    n,b = np.shape(D)

    sumTot = np.nansum(D)
    nbTot = int(n*(n-1)/2)

    nbCoin,b = np.shape(sameCoin)

    sumDistCoin = 0
    for ensemble in sameCoin:
        sumDistCoin = sumDistCoin + D[tuple(ensemble)]

    moyenneDistCoin = sumDistCoin/nbCoin
    moyenneDistAutre = (sumTot-sumDistCoin)/(nbTot-nbCoin)

    moyenneTot = sumTot/nbTot

    return (moyenneDistCoin - moyenneDistAutre)/moyenneTot
