#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 11:37:31 2020

@author: filipe
"""
import numpy as np

class EX2_normalize(object):
    def __init__(self):
        pass
    def calc(self,x):
        x=x/np.sum(np.power(x,2))
        return x
    
class Unitary(object):
    def __init__pass(self):
        pass
    def calc(self,x):
        return x