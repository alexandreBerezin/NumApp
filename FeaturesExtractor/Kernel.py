import numpy as np


class Kernel:
    '''classe representant la fonction de covariance'''
    
    def __init__(self,weights:np.ndarray,Xvalues:np.ndarray,l:float)->None:
        '''Constructeur du Kernel
        weights     : Vecteur 1D
        Xvalues     : Vecteur avec les valeurs des coordonnées [[xA,yA], [xB,yB], ...]
        l           : longeur caractéristique du RBF
        '''
        self.Xvalues = Xvalues
        self.weights = weights
        self.l = l
        self.N, = np.shape(weights)
    
    def __repr__(self):
        return "Objet Kernel : l = " + str(self.l) + ", N = " + str(self.N) 
    
    
    def getVarVector(self)->np.ndarray:
        '''Renvoie un vecteur 1D : variance pondéré de chaque pixel '''
        klZero = np.square(self.Xvalues)
        #somme de x^2 et y^2 
        klZero = np.sum(klZero,axis=1)
        klZero = np.exp(-klZero/(2*(self.l)**2))
        
        return np.convolve(klZero,self.weights)[:self.N]
    
    
    def cov(self,idxX:np.ndarray,idxY:np.ndarray)->float:
        '''Renvoie la covariance pondérée entre x et y'''
        sum = 0
        x = self.Xvalues[idxX]
        y = self.Xvalues[idxY]
        for idx in range(self.N):
            z = self.Xvalues[idx]
            sum = sum + self.RBF(x,z,self.l/2)*self.weights[idx]*self.RBF(z,y,self.l/2)
        return sum

    
    def RBF(self,A:np.ndarray,B:np.ndarray,l:float)->float:
        '''Renvoie la covariance par le kernel RBF entre A = [xA,yA] et B=[xB,yB]'''
        xA,yA = A
        xB,yB = B
        d2 = (xB-xA)**2 + (yB-yA)**2
        return np.exp(-d2/(2*l**2))
        
        
        
        
        