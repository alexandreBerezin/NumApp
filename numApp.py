'''
Fichier qui décrit les fonctions principales de l'application NumApp
'''

#importation des differents paquets
import FeaturesExtractor.main as fe
import ImageProcessing.processing as pr
import FeatureMatching.distance as fm

import multiprocessing as mp


import numpy as np


def getDistance(param:dict,path1:str,path2:str,debug=False)->float:

    alpha = param["alpha"]
    reprojThreshold = param["reprojThreshold"]
    denoiseTVWeight = param["denoiseTV weight"]
    
    #On récupère le nom
    name1 = (path1.split("/")[1])[:-4]
    name2 = (path2.split("/")[1])[:-4]
    
    if debug : print("Image processing")
    #image processing
    img1 =pr.cropToCoin(path1)
    img2 =pr.cropToCoin(path2)
    contours1 = pr.getContour(img1,denoiseTVWeight)
    contours2 = pr.getContour(img2,denoiseTVWeight)

    if debug : print("Calcul des points d'interets")
    #Calcul des features
    feat1 = fe.getFeatures(param,name1,contours1,debug)
    feat2 = fe.getFeatures(param,name2,contours2,debug)


    if debug : print("Calcul des matchs")
    #Calcul des match
    kp1,kp2,match = fm.getFilteredMatch(img1,feat1,img2,feat2,reprojThreshold)

    #Calcul de la distance
    [nb,p] = fm.getNbAndP(kp1,kp2,match)

    return -(nb-alpha*np.log(p))



def getFeaturesList(param,pathList):

    denoiseTVWeight = param["denoiseTV weight"]
    
    
    
    paramList = []
    for path in pathList:
        name = (path.split("/")[1])[:-4]
        img =pr.cropToCoin(path)
        contours = pr.getContour(img,denoiseTVWeight)
        paramList.append([param,name,contours])
        
        
    print("Start pool")
    with mp.Pool(4) as pool:
        L = pool.starmap(fe.getFeatures, paramList)
        
    '''
    nbChunk = 6
    chunks = [paramList[x:x+nbChunk] for x in range(0, len(paramList), nbChunk)]
    
    
    
    for paramChunk in chunks:
        print("Start pool")
        with mp.Pool() as pool:
            L = pool.starmap_async(fe.getFeatures, paramChunk)
    ''' 



    

def getDistanceMatrix(param:dict,pathList:list)->np.ndarray:
    n = len(pathList)
    D = np.empty((n,n))
    D.fill(np.nan)
    
    
    featureList = getFeaturesList(param,pathList)


    print("Calcul de distance")
    count = 1
    for i in range(n):
        print(i,"/",n)
        for j in range(i):
            path1 = pathList[i]
            path2 = pathList[j]
            D[i,j]=getDistance(param,path1,path2,debug=False)
    
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
