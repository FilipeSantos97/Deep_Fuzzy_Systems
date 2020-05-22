#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 11:46:15 2020

@author: filipe
"""

import numpy as np 
import math
from .utils import density
from .utils import Normalize

R0=1-math.cos(math.pi/6)

class classmodel(object):
    def __init__(self,ex2f,Density,norm):
        self.K=0
        self.EX2_Fixed=ex2f
        self.EX2_Normalizor=norm
        self.density=Density
    def train(self,x):
        x=self.EX2_Normalizor.calc(x)#Normalization that makes E(X2)=1
        if self.K==0:
            self.Focal_points=np.array([x])
            self.Cloud_members=np.array([1])
            self.Radius=np.array([R0])
            self.Number_of_clouds=1
            self.K=1
            self.Global_mean=x
            if self.EX2_Fixed:
                self.EX2=1
                self.Local_EX2=np.array([1])
            else:
                self.EX2=np.sum(np.power(x,2))
                self.Local_EX2=np.array([self.EX2])
                
        else:
            self.K,self.Global_mean,self.EX2=Update_parameters(x,self.Global_mean,self.EX2,self.K,self.EX2_Fixed)
             #var(x)=E(X2)-E(X)2
            FocalDensity=np.zeros(self.Number_of_clouds)
            for i in range(0,self.Number_of_clouds):
                FocalDensity[i]=self.density.calc(self.Focal_points[i],self.Global_mean,self.EX2)               
            MaxFD=max(FocalDensity)
            MinFD=min(FocalDensity)
            SampleDensity=self.density.calc(x,self.Global_mean,self.EX2) 
            if SampleDensity<MinFD or SampleDensity>MaxFD:
                self.Create_new_cloud(x)
            else:
                FocalDistance=np.zeros(self.Number_of_clouds)
                for i in range(0,self.Number_of_clouds):
                    FocalDistance[i]=Distance(x,self.Focal_points[i])
                Closest_cloud=np.argmin(Distance)            
                Closest_distance=FocalDistance[Closest_cloud]
                if Closest_distance>self.Radius[Closest_cloud]:
                    self.Create_new_cloud(x)
                else:
                    self.Update_cloud(x,Closest_cloud)
    
    def Create_new_cloud(self,x):
        self.Focal_points=np.append(self.Focal_points,[x],axis=0)
        self.Radius=np.append(self.Radius,R0)
        self.Cloud_members=np.append(self.Cloud_members,1)
        self.Number_of_clouds=self.Number_of_clouds+1
        if self.EX2_Fixed:
            self.Local_EX2=np.append(self.Local_EX2,[1])
        else:
            self.Local_EX2=np.append(self.Local_EX2,[np.sum(np.power(x,2))])
                
        
    def Update_cloud(self,x,Closest_cloud):
        self.Cloud_members[Closest_cloud],self.Focal_points[Closest_cloud],self.Local_EX2[Closest_cloud]=Update_parameters(x,self.Focal_points[Closest_cloud],self.Local_EX2[Closest_cloud],self.Cloud_members[Closest_cloud],self.EX2_Fixed)
        self.Radius[Closest_cloud]=np.sqrt(0.5*(self.Radius[Closest_cloud]**2+(self.Local_EX2[Closest_cloud]-np.sum(np.power(self.Focal_points[Closest_cloud],2)))))
    def get_Focal_points(self):
        return self.Focal_points,self.Number_of_clouds

class ALMMo0(object):
    def __init__(self,**kwargs):
        self.Classnumber=0
        self.models=[]
        self.density_def=density.Unimodal_Density()
        self.EX2_Normalizor=Normalize.Unitary()
        self.EX2_Fixed=False
        #usageof kwargs
        allowed_keys=set(['density_def','EX2_Normalizor'])
        for key, value in kwargs.items():
            if key in allowed_keys:
                self.__dict__.update([(key, value)])
                if key=='EX2_Normalizor':
                    self.EX2_Fixed=True
    
    def train(self,x,y):

        size=len(x)
        CL=np.max(y)+1;
        for i in range(0,CL):
            self.models.append(classmodel(self.EX2_Fixed,self.density_def,self.EX2_Normalizor))     
        for i in range (size-1,-1,-1):

            print("Training sample no.",size-i,"of",size)
            self.models[y[i]].train(x[i])

        self.Classnumber=CL # Number of existing classes

    def predict(self,x): #ALMMo-0 system for learning
        size=len(x)
        Prediction=np.zeros(size)
        for i in range(0,size):
            print("Predictinging sample no.",i,"of",size)
            x[i]=self.EX2_Normalizor.calc(x[i])
            T=np.zeros(self.Classnumber)
            for j in range (0,self.Classnumber):
                focal_point,nclouds=self.models[j].get_Focal_points()
                FocalDistance=np.zeros(nclouds)
                for k in range(0,nclouds):
                    FocalDistance[k]=Distance(x[i],focal_point[k])
                T[j]=min(FocalDistance)
            
            Prediction[i]=np.argmin(T)
                
        return Prediction
        

def Distance(x,y):
    Distance=np.sqrt(np.sum(np.power(x-y,2)))
    return Distance

def Update_parameters(x,mean,ex2,members,ex2_fixed):
    members=members+1
    mean=(mean*(members-1)+x)/members
    if ex2_fixed:
        ex2=1
    else:
        ex2=(ex2*(members-1)+np.sum(np.power(x,2)))/members
    return members,mean,ex2
