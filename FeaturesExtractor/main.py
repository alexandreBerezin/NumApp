#### Fichier main.py va encapsuler 
#### les fonction du package FeatureExtractor
import os
from matplotlib import image

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse as s
import cv2

import ImageProcessing.processing as pr
import FeaturesExtractor.Kernel as k
import FeaturesExtractor.Features as f



def computeFeatures(param:dict,path:str)->list:
    '''
    return [img,coordPI]
    --------------------
    fonction qui va encapsuler la recherche 
    de points d'interêts
    '''

    
    l = param["longeur RBF"]
    nbFeatures = param["nombre features"]
    denoiseTVWeight = param["denoiseTV weight"]

    #Utilisation de ImageProcessing
    img =pr.cropToCoin(path)
    contours = pr.getContour(img,denoiseTVWeight)

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


def getFeatures(param:dict,path:str)->list:
    '''
    return [img,coordPI]
    --------------------
    fonction qui va renvoyer les points
    d'interêts et les enregistrer dans
    un fichier 
    '''

    
    imgPath = path
    l = param["longeur RBF"]
    nbFeatures = param["nombre features"]
    denoiseTVWeight = param["denoiseTV weight"]
    
    name = "F_(" + imgPath[6:-4] + ")_l_%f_nb_%d_w_%f"%(l,nbFeatures,denoiseTVWeight)
    
    img =pr.cropToCoin(imgPath)

    ## Chercher si il existe une valeur déja calculé de K 
    basepath = 'Features/'
    

    for entry in os.listdir(basepath):
        if os.path.isfile(os.path.join(basepath, entry)):
            if(name in entry):
                features =  np.load(basepath + entry)
                return [img,features]
    

    [img,features] = computeFeatures(param,path)
    np.save(basepath + name, features)
    
    return [img,features]


