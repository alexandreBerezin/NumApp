import numpy as np
from scipy import sparse as s
import os

from FeaturesExtractor.misc import getCoordFromVect,getIdxFromArray,getCoordFromVectList


def getVoisins(idx:int,nbSide:int,d:int):
    '''
    renvoie la liste des indices voisins de idx
    ----------------------
    idx: index du pixel dans un vecteur 1D
    nbSide: nombre de pixel de coté 
    d : distance du plus grand voisins '''
    
    #Coordonnées du centre
    Cx,Cy = getCoordFromVect(idx,nbSide)
    #Coordonnées de l'origine
    Ox = Cx-d
    Oy = Cy-d
    voisins = []
    for x in range(Ox,Ox+2*d+1):
        for y in range(Oy,Oy+2*d+1):

            # Valeurs dans la grille ? 
            if ((x>=0 and x<nbSide) and (y>=0 and y <nbSide)):
                voisins.append([x,y])
                
    
    ## Tranformation en idices 1D
    nb = len(voisins)
    idxVoisins = np.zeros(nb)
    for coord in range(nb):
        i,j = voisins[coord]
        idxVoisins[coord] = getIdxFromArray(i,j,nbSide)
    
    return idxVoisins




def RBF(a:np.ndarray,b:np.ndarray,l:float):
    xa,ya = a
    xb,yb = b
    d2 = (xa-xb)**2 + (ya-yb)**2
    return np.exp(-d2/(2*l**2),dtype=np.float32)




def computeK(nbSide:int,l:float)-> s.lil.lil_matrix:
    '''
    Calcule et renvoie la matrice K approchée RBF
    avec un cube de longeur 6l pour chaque 
    ------------------------------
    nbSide: nombre de pixel de coté 
    l: longeur caractéristique du RBF
    '''
    d = int(np.floor(3*l))
    N = nbSide*nbSide
    M = s.lil_matrix((N,N),dtype=np.float32)
    

    for i in range(N):
        voisins = getVoisins(i,nbSide,d)
        x = getCoordFromVect(i,nbSide)
        for j in voisins:
            y = getCoordFromVect(j,nbSide)
            M[i,j] = RBF(x,y,l)
            
    return s.csc_matrix(M).astype(np.float32,copy=False)




def getK(nbSide:int,l:float,absPath:str)-> s.lil.lil_matrix:
    '''
    Renvoie la matrice approchée de K : 
    vérifie dans le dossier et sinon la calcule 
    et l'enregistre voir computeK 
    '''
    ## Chercher si il existe une valeur déja calculé de K 
    
    absKFolderPath = os.path.join(absPath,"Kdata/")

    for entry in os.listdir(absKFolderPath):
        if os.path.isfile(os.path.join(absKFolderPath, entry)):
            if("%f"%l in entry):
                path = absKFolderPath + entry
                return s.load_npz(path).astype(np.float32)
    
    print("Calcul de K")
    K = computeK(nbSide,l)
    print("sauvegarde de K")
    s.save_npz(absKFolderPath + '/L_%f'%l, K)
    return K



def getKw(weightVec:np.ndarray,nbSide:int,l:float,absPath:str):
    '''
    Renvoie la matrice de covariance approchée pondéré par
    les poids du vecteur
    '''



    K = getK(nbSide,l/2,absPath)
    W = s.diags(weightVec,dtype =np.float32)
 
    
    return K.transpose().dot(W).dot(K)
