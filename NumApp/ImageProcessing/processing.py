import numpy as np
import cv2
import matplotlib.pyplot as plt

from skimage.restoration import denoise_tv_chambolle


def preprocess(img :np.ndarray,flou:int,resize:int=0)->np.ndarray:
    '''Renvoie une image apres
    - flou gaussien
    - detecteur de contours (Laplace)'''

    img = cv2.GaussianBlur(img,(flou,flou),cv2.BORDER_DEFAULT)
    contours = cv2.convertScaleAbs(cv2.Laplacian(img,cv2.CV_64F))
    if(resize != 0):
        contours = cv2.resize(contours, (resize,resize), interpolation = cv2.INTER_AREA)
    contours = np.float32(contours)
    return contours
    
    
def cropToCoin(imgPath:str)-> np.ndarray:
    #Lecture
    img = cv2.imread(imgPath,cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    #Couper le bas 100 pixel
    deltaL = 100
    img = img[:-deltaL,:]
    
    #Convesrion en gris
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret,thresh1 = cv2.threshold(gray,253,255,cv2.THRESH_BINARY)
    
    #Calcul du centre de masse
    mass_x, mass_y = np.where(thresh1 <= 0)
    cent_x = np.average(mass_x)
    cent_y = np.average(mass_y)
    
    ### Recadrage
    SidePixel = 360
    pixG = int(cent_x - SidePixel/2)
    pixD = int(cent_x + SidePixel/2)
    pixH = int(cent_y - SidePixel/2)
    pixB = int(cent_y + SidePixel/2)
    
    return img[pixG:pixD,pixH:pixB]
    
    
def getContour(img:np.ndarray,denoiseTVWeight:float)->np.ndarray:
    #transformation en nuances de gris 
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Total variation regularization
    denoisedImg = denoise_tv_chambolle(gray,weight=denoiseTVWeight)

    #CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    denoisedImg = cv2.normalize(src=denoisedImg, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cl1 = clahe.apply(denoisedImg)
    
    #Laplace
    Laplace = cv2.Laplacian(cl1,cv2.CV_64F)
    contours = cv2.convertScaleAbs(Laplace)

    #Normalisation
    norm = np.linalg.norm(contours)
    contours = contours/norm
    
    return contours




