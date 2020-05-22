#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:33:14 2020

@author: filipe
"""
import tensorflow as tf
import numpy as np
import math
from .utils import density,Normalize



ETA0=0.1
OMEGA0=10

class cloud (object):
    def __init__(self):
        pass
    def Create_first_cloud(self,x,N,NC,ex2f):
        self.Focal_point=x
        self.Cloud_members=1
        self.EX2_Fixed=ex2f
        if self.EX2_Fixed:
            self.EX2=1
        else:
            self.EX2=np.sum(np.power(x,2))
        self.Past_NDensity_Sum=1
        self.Birth_Index=1
        self.A=[np.matrix([np.zeros(N+1)]) for i in range(NC)]
        self.C=np.eye(N+1)*OMEGA0

    def Create(self,x,N,K,new_A,ex2f):
        self.Focal_point=x
        self.EX2_Fixed=ex2f
        if self.EX2_Fixed:
            self.EX2=1
        else:
            self.EX2=np.sum(np.power(x,2))
        self.Cloud_members=1
        self.Past_NDensity_Sum=0
        self.Birth_Index=K
        self.A=new_A
        self.C=np.eye(N+1)*OMEGA0

    def Update(self,x):
        self.Cloud_members,self.Focal_point,self.EX2=Update_parameters(x,self.Focal_point,self.EX2,self.Cloud_members,self.EX2_Fixed)
    def Rewrite(self,x,K,N):
        self.Focal_point=(x+self.Focal_point)/2
        if self.EX2_Fixed:
            self.EX2=1
        else:
            self.EX2=(np.sum(np.power(x,2))+self.EX2)/2
        self.Cloud_members=math.ceil((self.Cloud_members+1)/2)
        self.Past_NDensity_Sum=0
        self.Birth_Index=K
        self.C=np.eye(N+1)*OMEGA0
    def Update_consequents(self,xe,NDensity,y,Number_of_classes):
        self.C=self.C-((self.C*xe.transpose()*xe*self.C)*NDensity)/(1+NDensity*xe*self.C*xe.transpose())
        for i in range(Number_of_classes):
            self.A[i]=self.A[i]+((self.C*xe.transpose()*(y[i]-xe*self.A[i].transpose())).transpose()*NDensity)
class ALMMo1(object):
    def __init__(self,**kwargs):
        self.K=0;
        self.Clouds=[]
        self.density_def=density.Unimodal_Density()
        self.type="Regressor"
        self.EX2_Normalizor=Normalize.Unitary()
        self.EX2_Fixed=False
        #usageof kwargs
        allowed_keys=set(['density_def','type','EX2_Normalizer'])
        allowed_types=["Regressor","Binary Classifier","Multiclass Classifier"]
        for key, value in kwargs.items():
            if key in allowed_keys:
                if key!='type':
                    self.__dict__.update([(key, value)])
                    if key=='EX2_Normalizor':
                        self.EX2_Fixed=True
                elif value in allowed_types:
                    self.__dict__.update([(key, value)])
                else:
                    raise SyntaxError('Invalid System Type')
                    
    def add_empty_cloud(self):
        self.Clouds.append(cloud())
    def calculate_sample_local_density(self,x):
        LocalDensity=np.zeros(self.Number_of_clouds)
        for i in range(0,self.Number_of_clouds):
            temp_members,temp_mean,temp_ex2=Update_parameters(x,self.Clouds[i].Focal_point,self.Clouds[i].EX2,self.Clouds[i].Cloud_members,self.EX2_Fixed)
            LocalDensity[i]=self.density_def.calc(x,temp_mean,temp_ex2)
        return LocalDensity
        
    def detect_stale_cloud(self):
        pass
    def train(self,x,y):
        M,N=x.shape
        if self.type=="Multiclass Classifier":
            y=tf.keras.utils.to_categorical(y)
        if self.K==0:
            x[0]=self.EX2_Normalizor.calc(x[0])
            self.K=1;
            self.Mean=x[0]
            if self.EX2_Fixed:
                self.EX2=1
            else:
                self.EX2=np.sum(np.power(x,2))
            self.Number_of_clouds=1
            if self.type=="Multiclass Classifier":
                M,self.Number_of_classes=y.shape
            else:
                self.Number_of_classes=1
            self.add_empty_cloud()
            self.Clouds[0].Create_first_cloud(x[0],N,self.Number_of_classes,self.EX2_Fixed)
        for i in range(1,M):
            print("training sample",i+1,"out of",M)
            x[i]=self.EX2_Normalizor.calc(x[i])
            #global parameters update
            self.K,self.Mean,self.EX2=Update_parameters(x[i],self.Mean,self.EX2,self.K,self.EX2_Fixed)
            #density calculation
            FocalDensity=np.zeros(self.Number_of_clouds)
            for j in range(0,self.Number_of_clouds):
                FocalDensity[j]=self.density_def.calc(self.Clouds[j].Focal_point,self.Mean,self.EX2) 
            MaxFD=max(FocalDensity)
            MinFD=min(FocalDensity)

            SampleDensity=self.density_def.calc(x[i],self.Mean,self.EX2)
            #first condition (density anomaly)
            if SampleDensity<MinFD or SampleDensity>MaxFD:
                
                LocalDensity=self.calculate_sample_local_density(x[i])
                if max(LocalDensity)>0.8:#ovelap exists
                    overlap=np.argmax(LocalDensity)
                    self.Clouds[overlap].Rewrite(x[i],self.K,N)

                else:

                    A_avg=[]
                    for j in range(self.Number_of_classes):
                        A_avg_line=0
                        for k in range(self.Number_of_clouds):
                            A_avg_line=A_avg_line+self.Clouds[k].A[j] 
                        A_avg_line=A_avg_line/self.Number_of_clouds
                        A_avg.append(A_avg_line)
                    self.Number_of_clouds=self.Number_of_clouds+1
                    self.add_empty_cloud()
                    self.Clouds[self.Number_of_clouds-1].Create(x[i],N,self.K,A_avg,self.EX2_Fixed)
            else: 

                dist=np.zeros(self.Number_of_clouds)
                for j in range(self.Number_of_clouds):
                    dist[j]=Distance(x[i],self.Clouds[j].Focal_point)
                ClosestCloud=np.argmin(dist)
                #Update existing cloud
                self.Clouds[ClosestCloud].Update(x[i])
            #detect stale clouds
            LocalDensity=self.calculate_sample_local_density(x[i])
            if np.sum(LocalDensity)==0:
                NDensity=np.ones(self.Number_of_clouds)/self.Number_of_clouds
            else:
                NDensity=LocalDensity/np.sum(LocalDensity)
##            print(NDensity)
            NC=self.Number_of_clouds
            for j in range(NC-1,-1,-1):
                self.Clouds[j].Past_NDensity_Sum=self.Clouds[j].Past_NDensity_Sum+NDensity[j]
                Utility=self.Clouds[j].Past_NDensity_Sum/(self.K-self.Clouds[j].Birth_Index)
                if self.Clouds[j].Birth_Index!=self.K and Utility<=ETA0:
                    del self.Clouds[j]
                    self.Number_of_clouds=self.Number_of_clouds-1

        #Update a C
            xe=np.append(np.matrix([[1]]),x[i],axis=1)

            for j in range(0,self.Number_of_clouds):
                self.Clouds[j].Update_consequents(xe,NDensity[j],y[i],self.Number_of_classes)
            print(self.Number_of_clouds)    
    def predict(self,x):
        L,W=x.shape
        Estimation=np.zeros([L,self.Number_of_classes])
        for i in range(L):
            x[i]=self.EX2_Normalizor.calc(x[i])
            xe=np.append(np.matrix([[1]]),x[i],axis=1)
            LocalDensity=self.calculate_sample_local_density(x[i])
            if np.sum(LocalDensity)==0:
                NDensity=np.ones(self.Number_of_clouds)/self.Number_of_clouds
            else:
                NDensity=LocalDensity/np.sum(LocalDensity)
            for j in range(self.Number_of_classes):
                for k in range(0,self.Number_of_clouds):
                    Estimation[i,j]=Estimation[i,j]+NDensity[k]*xe*self.Clouds[k].A[j].transpose()
                if self.type=="Binary Classifier":
                    Estimation[i,j]=round(Estimation[i])
        if self.type=="Multiclass Classifier":
            Estimation=np.argmax(Estimation, axis=1)
        
        return Estimation
        
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

###############################################################################
###############################################################################
def ALMMo1_learning(data,y):
   # print("training sample",1,"out of",len(data))
    K=1
    M,N=data.shape
    Global_mean=data[0]
    Global_EX2=np.sum(np.power(data[0],2))
    Focal_points=np.array(data[0])
    Cloud_members=np.array([1])
    Local_EX2=np.array([np.sum(np.power(data[0],2))])
    Number_of_clouds=1
    Past_NDensity_Sum=np.array([1])
    Birth_Index=np.array([1])
    A=np.array([np.zeros(N+1)])
    C=[np.eye(N+1)*10]
    for i in range(1,len(data)):
        #print("training sample",i+1,"out of",len(data))
        #global parameters update
        K=K+1
        Global_mean=(Global_mean*(K-1)+data[i])/K
        Global_EX2=(Global_EX2*(K-1)+np.sum(np.power(data[i],2)))/K
        #density calculation
        Var=Global_EX2-np.sum(np.power(Global_mean,2)) #var(x)=E(X2)-E(X)2
        FocalDensity=np.zeros(Number_of_clouds)
        for j in range(0,Number_of_clouds):
            FocalDensity[j]=1/(1+np.sum(np.power(Focal_points[j]-Global_mean,2))/Var) 
        MaxFD=max(FocalDensity)
        MinFD=min(FocalDensity)
        SampleDensity=1/(1+np.sum(np.power(data[i]-Global_mean,2))/Var)
        #first condition (density anomaly)
        if SampleDensity<MinFD or SampleDensity>MaxFD:
            LocalDensity=np.zeros(Number_of_clouds)
            for j in range(0,Number_of_clouds):
                S=Cloud_members[j]
                Num=(S**2)*np.sum(np.power(data[i]-Focal_points[j],2));
                Denum=(S+1)*(S*Local_EX2[j]+np.sum(np.power(data[i],2)))-np.sum(np.power(data[i]+(S*Focal_points[j]),2))
                
                if (Num+Denum)==0 or Denum==0:
                    LocalDensity[j]=1
                else:
                    LocalDensity[j]=1/(1+Num/Denum)
            if max(LocalDensity)>0.8:#ovelap exists
                print(1,1)
                overlap=np.argmax(LocalDensity)
                new_FP= np.append(np.delete(Focal_points,overlap,axis=0),(data[j]+Focal_points[overlap])/2,axis=0)
                new_EX2=np.append(np.delete(Local_EX2,overlap,axis=0),(np.sum(np.power(data[j],2))+Local_EX2[overlap])/2)
                new_members=np.append(np.delete(Cloud_members,overlap,axis=0),math.ceil((Cloud_members[overlap]+1)/2))
                new_Past=np.append(np.delete(Past_NDensity_Sum,overlap,axis=0),0)
                new_Birth=np.append(np.delete(Birth_Index,overlap,axis=0),K)
                new_A=np.append(np.delete(A,overlap,axis=0),np.array([A[overlap]]),axis=0)
                new_C=C
                del new_C[overlap]
                new_C=new_C+[np.eye(N+1)*10]
                

                
                
            else:
                print(1,0)
                Number_of_clouds=Number_of_clouds+1
                Focal_points=np.append(Focal_points,data[i],axis=0)
                Local_EX2=np.append(Local_EX2,np.sum(np.power(data[i],2)))
                Cloud_members=np.append(Cloud_members,[1])
                Past_NDensity_Sum=np.append(Past_NDensity_Sum,[0])
                Birth_Index=np.append(Birth_Index,K)
                new_A_line=np.sum(A,axis=0)/Number_of_clouds
                A=np.append(A,[new_A_line],axis=0)
                C=C+[np.eye(N+1)*10]
        else:
            print(0)
            dist=np.zeros(Number_of_clouds)
            for j in range(0,Number_of_clouds):
                dist[j]=np.sqrt(np.sum(np.power(data[i]-Focal_points[j],2)))
            ClosestCloud=np.argmin(dist)

            #Update existing cloud
            Cloud_members[ClosestCloud]=Cloud_members[ClosestCloud]+1
            Focal_points[ClosestCloud]=(Focal_points[ClosestCloud]*(Cloud_members[ClosestCloud]-1)+data[i])/Cloud_members[ClosestCloud]
            Local_EX2[ClosestCloud]=(Local_EX2[ClosestCloud]*(Cloud_members[ClosestCloud]-1)+np.sum(np.power(data[i],2)))/Cloud_members[ClosestCloud]
        LocalDensity=np.zeros(Number_of_clouds)
        for j in range(0,Number_of_clouds):
            S=Cloud_members[j]
            Num=(S**2)*np.sum(np.power(data[i]-Focal_points[j],2));
            Denum=(S+1)*(S*Local_EX2[j]+np.sum(np.power(data[i],2)))-np.sum(np.power(data[i]+(S*Focal_points[j]),2))
            if (Num+Denum)==0:
                LocalDensity[j]=1
            else:
                LocalDensity[j]=1/(1+Num/Denum)
        if np.sum(LocalDensity)==0:
            NDensity=np.ones(Number_of_clouds)/Number_of_clouds
        else:
            NDensity=LocalDensity/np.sum(LocalDensity)

        Present_NDensity_Sum=Past_NDensity_Sum+NDensity
        Keep=[];
        sth_removed=0
        for j in range(0,Number_of_clouds):
            if Birth_Index[j]==K:
                Keep=Keep+[j]
            elif Present_NDensity_Sum[j]/(K-Birth_Index[j])>ETA0:
                Keep=Keep+[j]
            else:
                sth_removed=sth_removed+1

        if sth_removed:
            new_FP= Focal_points[Keep]
            new_EX2=Local_EX2[Keep]
            new_members=Cloud_members[Keep]
            new_Past=Present_NDensity_Sum[Keep]
            new_Birth=Birth_Index[Keep]
            new_A=A[Keep]
            new_C=[C[m] for m in Keep]
            Focal_points=new_FP
            Local_EX2=new_EX2
            Cloud_members=new_members
            Past_NDensity_Sum=new_Past
            Birth_Index=new_Birth
            A=new_A
            C=new_C
            Number_of_clouds=Number_of_clouds-sth_removed
        else:
            Past_NDensity_Sum=Present_NDensity_Sum
        
        xe=np.append(np.array([[1]]),data[i],axis=1)
        for ii in range(0,Number_of_clouds):
            C[ii]=C[ii]-((C[ii]*xe.transpose()*xe*C[ii])*NDensity[ii])/(1+NDensity[ii]*xe*C[ii]*xe.transpose())
            A[ii]=A[ii]+((C[ii]*xe.transpose()*(y[i]-xe*np.array([A[ii]]).transpose())).transpose()*NDensity[ii])
           
    SysP={}
    SysP["Focal points"]=Focal_points
    SysP["Cloud members"]=Cloud_members
    SysP["Local EX2"]=Local_EX2
    SysP["Number of clouds"]=Number_of_clouds
    SysP["K"]=K
    SysP["Global mean"]=Global_mean
    SysP["Global EX2"]=Global_EX2
    SysP["Birth Index"]=Birth_Index
    SysP["Past NDensity Sum"]=Past_NDensity_Sum
    SysP["A"]=A
    SysP["C"]=C
    return SysP

def ALMMo1_regressor_testing(data,SysP):
    # Calculate the system output
    Estimation=np.zeros(len(data))
    for i in range(len(data)):
        xe=np.append(np.array([[1]]),data[i],axis=1)
        for ii in range(0,SysP["Number of clouds"]):
            LocalDensity=np.zeros(SysP["Number of clouds"])
            for j in range(0,SysP["Number of clouds"]):
                S=SysP["Cloud members"][j]
                Num=(S**2)*np.sum(np.power(data[i]-SysP["Focal points"][j],2));
                Denum=(S+1)*(S*SysP["Local EX2"][j]+np.sum(np.power(data[i],2)))-np.sum(np.power(data[i]+(S*SysP["Focal points"][j]),2))
                if (Num+Denum)==0:
                    LocalDensity[j]=1
                else:
                    LocalDensity[j]=1/(1+Num/Denum)
            if np.sum(LocalDensity)==0:
                NDensity=np.ones(SysP["Number of clouds"])/SysP["Number of clouds"]
            else:
                NDensity=LocalDensity/np.sum(LocalDensity)
            Estimation[i]=Estimation[i]+NDensity[ii]*xe*np.array([SysP["A"][ii]]).transpose()
    return Estimation

def ALMMo1_binary_classifier_testing(data,SysP):
    # Calculate the system output
    Estimation=np.zeros(len(data))
    for i in range(len(data)):
        xe=np.append(np.array([[1]]),data[i],axis=1)
        for ii in range(0,SysP["Number of clouds"]):
            LocalDensity=np.zeros(SysP["Number of clouds"])
            for j in range(0,SysP["Number of clouds"]):
                S=SysP["Cloud members"][j]
                Num=(S**2)*np.sum(np.power(data[i]-SysP["Focal points"][j],2));
                Denum=(S+1)*(S*SysP["Local EX2"][j]+np.sum(np.power(data[i],2)))-np.sum(np.power(data[i]+(S*SysP["Focal points"][j]),2))
                if (Num+Denum)==0:
                    LocalDensity[j]=1
                else:
                    LocalDensity[j]=1/(1+Num/Denum)
            if np.sum(LocalDensity)==0:
                NDensity=np.ones(SysP["Number of clouds"])/SysP["Number of clouds"]
            else:
                NDensity=LocalDensity/np.sum(LocalDensity)
            Estimation[i]=Estimation[i]+NDensity[ii]*xe*np.array([SysP["A"][ii]]).transpose()
        Estimation[i]=round(Estimation[i])
    return Estimation