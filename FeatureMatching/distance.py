'''
Fichier distance

ensemble de fonctions qui permettent le calcul de distances

'''

import cv2
import numpy as np

from scipy.spatial import procrustes

def getFilteredMatch(img1,features1,img2,features2):
    '''
    renvoie une liste de match Par Brute Force
    et descripteur BRIEF    
    '''
    keyPoint1 = []

    for coord in features1:
        x,y = coord
        keyPoint1.append(cv2.KeyPoint(int(x),int(y),1))
        
    keyPoint2 = []

    for coord in features2:
        x,y = coord
        keyPoint2.append(cv2.KeyPoint(int(x),int(y),1))
        
    
    orb = cv2.ORB_create()

    keyPoint1,des1 = orb.compute(img1,keyPoint1)
    keyPoint2,des2 = orb.compute(img2,keyPoint2)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
    matches = bf.match(des1,des2)
    
    matches = sorted(matches,key=lambda x:x.distance)
    
    src_pts = np.float32([ keyPoint1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ keyPoint2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5)

    MatchesF = np.extract(mask.ravel(),matches)
    
    return [keyPoint1,keyPoint2,MatchesF]

def getNbAndP(keyPoint1,keyPoint2,match):
    src_pts = np.float32([ keyPoint1[m.queryIdx].pt for m in match ]).reshape(-1,2)
    dst_pts = np.float32([ keyPoint2[m.trainIdx].pt for m in match ]).reshape(-1,2)
    
    a,b,p=procrustes(src_pts,dst_pts)
    
    nb = np.shape(match)[0]
    
    
    return [nb,p]
    
    
