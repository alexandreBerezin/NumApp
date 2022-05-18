import numpy as np
from FeaturesExtractor import Kernel

from misc import *
    

    
class Extractor:
    def __init__(self,varVector:np.ndarray,kernel:Kernel) -> None:
        self.featuresList = []
        self.varVector = varVector
        self.N, = np.shape(varVector)
        self.kernel = kernel
        self.psi = np.array([])
        
        
    def getIndexFirstFeature(self)->int:
        '''Renvoie le premier index de la feature
        '''
        idx = np.argmax(self.varVector)
        return idx #car 1D Array
    
    def getIndexNextFeature(self,varVect:np.ndarray)->int:
        '''Renvoie l'index de la prochaine feature'''
        idx = np.argmax(varVect)
        return idx
        
        
        

    def addFeature(self,idx):
        '''Add a feature and update psi matrix'''
        
        print("Ajout de Feature :")
        print("liste : ",self.featuresList)
        
        ##Si il n'y a pas encore de features
        if (self.featuresList == []):
            #On ajoute la première
            self.featuresList.append(idx)
            #On fait la mise a jour
            covXsiX = self.kernel.covWithX(idx)
            self.psi = np.reshape(covXsiX,(-1,1))
        else : 
            self.featuresList.append(idx)
            covXsiX = self.kernel.covWithX(idx)
            covXsiX = covXsiX.reshape(-1,1)
            self.psi = np.append(self.psi,covXsiX,axis = 1)


    def getVarVector(self)->np.ndarray:
        print("Utilisation de Var Vect2")
        xsiList = np.array(self.featuresList)
        l = np.size(xsiList)
        xsiCov = np.zeros_like(xsiList)
        for idx in range(l):
            x = xsiList[idx]
            xsiCov[idx] = self.kernel.cov(x,x,Usetreshold=False)
        
        invXsiCov = 1/xsiCov
        
        invXsiCovDiag = np.diag(invXsiCov)
        
        A = self.varVector - (self.psi).dot(invXsiCovDiag).dot(self.psi.T)
        

        return np.diag(A)
        
    
    
    def getFeature(self,nbFeatures:int)->list:
        '''renvoie la liste des nb features de l'image'''
        
        FF = self.getIndexFirstFeature()
        if nbFeatures ==1 :
            return FF
        self.addFeature(FF)
        
        for i in range(1,nbFeatures):
            newVarVec = self.getVarVector()
            newFeature = self.getIndexNextFeature(newVarVec)
            self.addFeature(newFeature)
            
        return self.featuresList
            
    
    
        
