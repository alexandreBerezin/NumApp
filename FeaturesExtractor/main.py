#### Fichier main.py va encapsuler 
#### les fonction du package FeatureExtractor
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse as s
import cv2

import ImageProcessing.processing as pr
import FeaturesExtractor.Kernel as k
import FeaturesExtractor.Features as f



def getFeatures(param:dict)->list:
    '''
    return [img,coordPI]
    --------------------
    fonction qui va encapsuler la recherche 
    de points d'interêts
    '''

    imgPath = param["chemin"]
    l = param["longeur caractéristique du RBF"]
    nbFeatures = param["nombre de points d'interêts"]

    #Utilisation de ImageProcessing
    img =pr.cropToCoin(imgPath)
    contours = pr.getContour(img)

    shape = np.shape(contours)
    nbSide,b = shape

    #Transformation en vecteur 1D
    weightVec = np.ravel(contours)

    #Calcul de K
    Kw = k.getKw(weightVec,nbSide,l)
    ## Extracteur
    ext = f.Extractor(Kw)
    #
    coordPI= ext.getCoordNFeatures(nbFeatures,nbSide)
    nb,b = np.shape(coordPI) 


    ## Conversion en X Y 
    for i in range(nb):
        y,x = coordPI[i]
        coordPI[i] = np.array([x,y])



    return [img,coordPI]
