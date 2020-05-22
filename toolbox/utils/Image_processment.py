#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 11:37:31 2020

@author: filipe
"""
from scipy import ndimage
from skimage import transform
import numpy as np

def rescale(img,h,w):
    
    img=transform.resize(img, (h, w))    
    return(img)
    
def rotate(img,ang):

    img = ndimage.rotate(img, ang)
    return(img)

def rotateset(imset,ang):
    num_img=len(imset)
    img=rotate(imset[0],ang)
    ii,jj=img.shape
    rotated_set=np.ones([num_img,ii,jj])
    rotated_set[0]=img
    for i in range(1,len(imset)):
        rotated_set[i]=rotate(imset[i],ang)
    return rotated_set

def rescaleset(imset,h,w):
    
    num_img=len(imset)
    rescaled_set=np.ones([num_img,h,w])
    for i in range(1,len(imset)):
        rescaled_set[i]=rescale(imset[i],h,w)
    return rescaled_set

def flattenset(imset):
    
    num_img,ii,jj=imset.shape
    flattened_set=np.ones([num_img,ii*jj])
    for i in range(len(imset)):
        flattened_set[i]=imset[i].flat
    return flattened_set

