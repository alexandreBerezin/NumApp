import numpy as np
from scipy import sparse as s
import os

from FeaturesExtractor.misc import getCoordFromVect,getIdxFromArray,getCoordFromVectList






class Extractor:
    '''
    Class qui est utilisée pour récupérer les points d'interêts 
    '''
    def __init__(self,Kw:np.ndarray):
        self.featuresList = []
        self.KwBase = Kw
        self.varVectBase = Kw.diagonal()
        self.varVect = self.varVectBase
        self.psi = []
        
    def getIndexFirstFeature(self):
        idx = np.argmax(self.varVectBase)
        return idx
    
    def getIndexNextFeature(self)->int:
        '''Renvoie l'index de la prochaine feature'''
        idx = np.argmax(self.varVect)
        return idx
    
    def addFeature(self,idx:int)->None:
        '''On ajoute la feature et on fait la mise à jour de psi '''
        
        self.featuresList.append(idx)
        ## Si aucune feature on ajoute la nouvelle
        if self.psi == []:
            self.psi = self.KwBase[:,idx]
        else:
            self.psi = s.hstack([self.psi,self.KwBase[:,idx]])
        
        
    def updateVarVect(self):
        '''Add a feature and update varVect'''
        
        #Calcul de 
        varXsi = self.varVectBase[self.featuresList]

        invVarXsi = 1/varXsi
        invVarXsiDiag = s.diags(invVarXsi)
                        
        self.varVect = s.csc_matrix(self.varVectBase) - (self.psi.dot(invVarXsiDiag).dot(self.psi.transpose())).diagonal()
        
        
    def getVarVect(self):
        return self.varVect


    def getNFeatures(self,nbFeatures:int)-> list:
        ### Add first feature
        idx = self.getIndexFirstFeature()
        self.addFeature(idx)
        self.updateVarVect()

        for i in range(1,nbFeatures):
            print("features: ",i,'/',nbFeatures)
            idx = self.getIndexNextFeature()
            self.addFeature(idx)
            self.updateVarVect()

        return self.featuresList
        
        
    
    def getCoordNFeatures(self,nbFeatures:int,nbSide:int)->np.ndarray:
        liste = self.getNFeatures(nbFeatures)
        return getCoordFromVectList(liste,nbSide)
        