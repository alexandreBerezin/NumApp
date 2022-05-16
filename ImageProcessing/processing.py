import numpy as np
import cv2
import matplotlib.pyplot as plt


def preprocess(imgPath:str,size:int)->np.ndarray:
    '''Renvoie une image après
    - flou gaussien
    - detecteur de contours (Laplace)'''
    
    img = cv2.imread(imgPath,0)
    
    contours = cv2.convertScaleAbs(cv2.Laplacian(img,cv2.CV_64F))
    contours = cv2.resize(contours, (size,size), interpolation = cv2.INTER_AREA)
    contours = np.float32(contours)
    return contours
    


