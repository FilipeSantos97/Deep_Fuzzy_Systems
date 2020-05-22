#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 14:42:20 2020

@author: filipe
"""
import numpy as np

class Unimodal_Density(object):
    
    def __init__(self):
        
        pass
    
    def calc(self,x,mean,ex2):
        
        var=ex2-np.sum(np.power(mean,2))
        if var==0:
            result=1
        else:
            result=1/(1+np.sum(np.power(x-mean,2))/var)          
        return result
    
class unitary(object):
    
    def __init__(self):
        pass 
    def calc(self,x,mean,ex2):
        return 1