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

    shape = np.shape(contours)
    nbSide,b = shape

    #Transformation en vecteur 1D
    weightVec = np.ravel(contours)
    
    id =np.random.randint(0,100)

    print("calcul Kw",id)
    #Calcul de K
    Kw = k.getKw(weightVec,nbSide,l)
    print("Ok Kw pour",id)
    ## Extracteur
    
    
    ext = f.Extractor(Kw)
    
    print("calcul features",id)
    coordPI= ext.getCoordNFeatures(nbFeatures,nbSide)
    nb,b = np.shape(coordPI) 
    print("Fin de calcul features ",id)

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
    
    if debug : print("getFeature pour ",nameCoin)
    
    
    #img =pr.cropToCoin(imgPath)

    ## Chercher si il existe une valeur déja calculé de K 
    basepath = 'Features/'
    

    for entry in os.listdir(basepath):
        if os.path.isfile(os.path.join(basepath, entry)):
            if(name in entry):
                features =  np.load(basepath + entry)
                return features
    

    features= computeFeatures(param,contours)
    np.save(basepath + name, features)
    
    return features


