#### Fichier main.py va encapsuler 
#### les fonction du package FeatureExtractor
import os
from matplotlib import image

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse as s
import cv2

import FeaturesExtractor.Kernel as k
import FeaturesExtractor.Features as f



def computeFeatures(param:dict,contours:np.ndarray)->list:
    '''
    return coordPI
    --------------------
    fonction qui va encapsuler la recherche 
    de points d'interêts
    '''

    
    l = param["longeur RBF"]
    nbFeatures = param["nombre features"]

    absPath = param["absPath"]

    shape = np.shape(contours)
    nbSide,b = shape

    #Transformation en vecteur 1D
    weightVec = np.ravel(contours)
    

    #Calcul de K
    Kw = k.getKw(weightVec,nbSide,l,absPath)

    ## Extracteur
    
    
    ext = f.Extractor(Kw)
    

    coordPI= ext.getCoordNFeatures(nbFeatures,nbSide)
    nb,b = np.shape(coordPI) 


    ## Conversion en X Y 
    for i in range(nb):
        y,x = coordPI[i]
        coordPI[i] = np.array([x,y])

    return coordPI





def getFeatures(param:dict,nameCoin:str,contours:np.ndarray,debug=True)->list:
    '''
    return [img,features]
    --------------------
    fonction qui va renvoyer les points
    d'interêts et les enregistrer dans
    un fichier 
    '''


    l = param["longeur RBF"]
    nbFeatures = param["nombre features"]
    denoiseTVWeight = param["denoiseTV weight"]
    
    name = "F_(" + nameCoin + ")_l_%f_nb_%d_w_%f"%(l,nbFeatures,denoiseTVWeight)



    ## Chercher si il existe une valeur déja calculé de K 
    FeaturePathLocal = param["Features Folder"]
    absPath = param["absPath"]


    basepath = os.path.join(absPath,FeaturePathLocal)



    for entry in os.listdir(basepath):
        if os.path.isfile(os.path.join(basepath, entry)):
            if(name in entry):
                features =  np.load(basepath + entry)
                return features
    
    print("Calcul feature for " + nameCoin)
    features= computeFeatures(param,contours)
    np.save(basepath + name, features)
    
    return features


