import numpy as np
from FeaturesExtractor import Kernel
    

    
class Extractor:
    def __init__(self,varVector:np.ndarray,kernel:Kernel) -> None:
        self.featuresList = []
        self.varVector = varVector
        self.N, = np.shape(varVector)
        self.kernel = kernel
        
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
        print("ADD feature")
        self.featuresList.append(idx)
    
    
    def getVarVector(self)->np.ndarray:
        
        newVarVec = np.zeros_like(self.varVector)
        covMatFeatures = self.covMatFeatures()
        invCovMatF = np.linalg.inv(covMatFeatures)
        for idx in range(self.N):
        
            covVec = self.covVecFeatures(idx)
            newVarVec[idx] = self.varVector[idx] - (covVec.T).dot(invCovMatF).dot(covVec)
            print(idx,"/",self.N)

        return newVarVec
    
    def covMatFeatures(self):
        '''Retourne la matrice de covariance des features'''
        m = len(self.featuresList)
        mat = np.zeros((m,m))
        for i in range(m):
            for j in range(m):
                idxX = self.featuresList[i]
                idxY = self.featuresList[j]
                mat[i,j] = self.kernel.cov(idxX,idxY)
        return mat
        
        
    def covVecFeatures(self,x):
        '''Retourne le vecteur de covariance entre x et les features '''
        m = len(self.featuresList)
        vec = np.zeros(m)
        for idx in range(m):
            y = self.featuresList[idx]
            vec[idx] = self.kernel.cov(x,y)
        
        return vec
    
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
            
    
    
        
