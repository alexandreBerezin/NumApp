'''
Fichier qui décrit les fonctions principales de l'application NumApp
'''

#importation des differents paquets
import FeaturesExtractor.main as fe
import ImageProcessing.processing as pr
import FeatureMatching.distance as fm

import multiprocessing as mp


import numpy as np
import os

def getDistance(param:dict,path1:str,path2:str,debug=False)->float:

    alpha = param["alpha"]
    reprojThreshold = param["reprojThreshold"]
    denoiseTVWeight = param["denoiseTV weight"]
    
    #On récupère le nom
    name1 = (path1.split("/")[-1])[:-4]
    name2 = (path2.split("/")[-1])[:-4]
    
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



def getFeaturesList(param,pathDataList):

    denoiseTVWeight = param["denoiseTV weight"]

    nbPool = param["Multiprocess Pool"]
    
    paramList = []
    for path in pathDataList:
        name = (path.split("/")[-1])[:-4]
        img =pr.cropToCoin(path)
        contours = pr.getContour(img,denoiseTVWeight)
        paramList.append([param,name,contours])
        

    print("Start pool with %d"%nbPool)
    with mp.Pool(nbPool) as pool:
        L = pool.starmap(fe.getFeatures, paramList)
        


    

def getDistanceMatrix(param:dict)->np.ndarray:



    param["absPath"]=os.path.dirname(os.path.abspath(__file__))
    absPath = param["absPath"]
    dataPath = param["Data Folder"]
    absDataPath = os.path.join(absPath,dataPath)


    #Liste de data
    dataList = sorted(os.listdir(absDataPath))

    #liste path data
    pathDataList = []
    for coin in dataList:
        pathDataList.append(os.path.join(absDataPath,coin))

    print(pathDataList[1])



    n = len(pathDataList)
    D = np.empty((n,n))
    D.fill(np.nan)
    
    
    featureList = getFeaturesList(param,pathDataList)






    print("Calcul de la matrice de distance ligne par ligne")
    count = 1
    for i in range(n):
        print(i,"/",n)
        for j in range(i):
            path1 = pathDataList[i]
            path2 = pathDataList[j]
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
