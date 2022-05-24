import numpy as np



def getCoordFromVect(idx:int,nbSide:int)-> np.ndarray:
    '''
    renvoie les coordonnées du pixel en fonction de 
    son indice dans un tableau 1D 
    --------------------------
    idx : index dand le tableau 1D
    nbSide : nombre de pixel sur le coté de l image carrée
    '''
    x = idx//(nbSide)
    y = idx%(nbSide)
    return np.array([x,y]) 
    
def getIdxFromArray(i:int,j:int,nbSide:int)->int:
    '''
    renvoie l'index du pixel dont les coordonnées sont 
    i et j 
    --------------------------
    i,j : coordonnées
    nbSide : nombre de pixel sur le coté de l image carrée
    '''
    return i*nbSide+ j 


def getCoordFromVectList(liste :list,nbSide:int)->np.ndarray:
    out = []
    for idx in liste:
        x,y = getCoordFromVect(idx,nbSide)
        out.append([x,y])
    
    return np.array(out)