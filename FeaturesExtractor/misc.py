import numpy as np

def XVecFromArray(array:np.ndarray)->np.ndarray:
    '''Renvoie un vecter numpy contenant ses coordonnées après ravel'''
    posVector = np.zeros((np.size(array),2))
    i=0
    for ix in np.ndindex(*array.shape):
        posVector[i] = np.array(ix)
        i= i+1 
    return posVector