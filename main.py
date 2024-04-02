# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 10:39:47 2024

@author: LP263296
"""

import pandas as pd
import math 

data = pd.read_csv("data_.csv",sep=';',dtype='float')


import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from models import Model, ModelFtD, FCNN, LEBNN

device='cpu'



import numpy as np
from scipy.spatial import ConvexHull
from quadprog import solve_qp

from tqdm import tqdm
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


data=torch.tensor(data.values,dtype=torch.float32)


"""
data format :

P; dP13; x; y; z; rx; ry; rz; fx; fy; fz; f_norm  

Objective : learn the link between force and displacement ie
    u <-> f
    
u : displacement + rotations of the last edge

f : force applied on the structure ie. contact force + pressure forces


dP13 define the contact plan, never used in the NN


"""

plan = True
    
if not plan:
    u=data[:,2:8] #x;y;z;rx;ry;rz
    f=data[:,[0,8,9,10]] #P;fx;fy;fz
    latent_size=4
if plan:    
    u=data[:,[2,4,6]] #x;z;ry
    f=data[:,[0,8,10]] #P;fx;fz
    latent_size=3

if True:
    u_mean = torch.mean(u,0)
    f_mean = torch.mean(f,0)

    u_std = torch.std(u,0)
    f_std = torch.std(f,0)
    
    u_scale=u_std[[1,1,2]]
    f_scale=f_std[[0,-1,-1]]
    
    u=(u-u_mean)/u_std[[1,1,2]]
    f=(f-f_mean)/f_std[[0,-1,-1]]
    U=u
    F=f
    
dP13=data[:,1]
p=data[:,0]


dataset =  torch.utils.data.TensorDataset(u,f)

train_set, val_set = torch.utils.data.random_split(dataset, [math.floor(len(dataset)*0.8), math.ceil(len(dataset)*0.2)])
batch_size=4
train_loader, val_loader = DataLoader(train_set, batch_size), DataLoader(val_set, batch_size)



"""
Proposed model : 
    
    f^ = enc_f(f) 
    u^ = enc_u(u) 

    f^ = latent_model(u^)

    f \approx dec_f(enc_f(f))
    u \approx dec_u(enc_u(u))

ie :
    
    f \approx dec_f(latent_model(enc_u(u)))
"""


hidden_size=128


fclat= Model(FCNN(latent_size,hidden_size,latent_size)   ) 

"""fclat= Model(LEBNN(latent_size,hidden_size*2,latent_size),
             enc_u=FCNN(u.size(-1),hidden_size,latent_size),
             dec_u=FCNN(latent_size,hidden_size,u.size(-1)),
             enc_f=FCNN(f.size(-1),hidden_size,latent_size),
             dec_f=FCNN(latent_size,hidden_size,f.size(-1)))
             """
            
             
fclatFtD= ModelFtD(FCNN(latent_size,hidden_size,latent_size)) 

             
             
            
def train(model, criterion = nn.L1Loss(), autoenc=False, epochs=100, lr=0.1, name=''):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    err_train=[]
    err_val=[]
    for epoch in tqdm(range(epochs)):
        err_train.append(0)
        for data in train_loader:
            optimizer.zero_grad()
            u, f = data
            u=u.to(device)
            f=f.to(device)
            
            fpred=model(u)
            
            err = criterion(fpred,f)
            err_train[-1]+=float(err)/len(train_loader)

            if autoenc:
                u_reconstruct = model.autoenc_u(u)
                f_reconstruct = model.autoenc_f(f)
                err = err + 0.1*criterion(u_reconstruct,u) + 0.1*criterion(f_reconstruct,f)
            err.backward()
            optimizer.step()
            #model.enc_f.weight.data=torch.linalg.inv(model.dec_f.weight.data)

        if min(err_train)==err_train[-1]:
            torch.save(model.state_dict(),name)
        if 1.5*min(err_train)<err_train[-1]:
            model.load_state_dict(torch.load(name))
        err_val.append(0)
        with torch.no_grad():
            for data in val_loader:
                u, f = data
                u=u.to(device)
                f=f.to(device)
                
                fpred=model(u)
                
                err = criterion(fpred,f)     

                err_val[-1]+=float(err)/len(val_loader)
                
    plt.plot(list(range(epochs)), err_train, list(range(epochs)), err_val)
    return err_train, err_val

def trainFtD(model, criterion = nn.L1Loss(), autoenc=False, epochs=100, lr=0.1, name=''):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    err_train=[]
    err_val=[]
    for epoch in tqdm(range(epochs)):
        err_train.append(0)
        for data in train_loader:
            optimizer.zero_grad()
            u, f = data
            u=u.to(device)
            f=f.to(device)
            
            upred=model.backward(f)
            
            err = criterion(upred,u)
            err_train[-1]+=float(err)/len(train_loader)

            if autoenc:
                u_reconstruct = model.autoenc_u(u)
                f_reconstruct = model.autoenc_f(f)
                err = err + 0.1*criterion(u_reconstruct,u) + 0.1*criterion(f_reconstruct,f)

            err.backward()
            optimizer.step()
            #model.dec_f.weight.data=torch.linalg.inv(model.enc_f.weight.data)

        if min(err_train)==err_train[-1]:
            torch.save(model.state_dict(),name)
        if 1.5*min(err_train)<err_train[-1]:
            model.load_state_dict(torch.load(name))
        err_val.append(0)
        with torch.no_grad():
            for data in val_loader:
                u, f = data
                u=u.to(device)
                f=f.to(device)
                
                upred=model.backward(f)
                
                err = criterion(upred,u)     

                err_val[-1]+=float(err)/len(val_loader)
                
    plt.plot(list(range(epochs)), err_train, list(range(epochs)), err_val)
    return err_train, err_val


if False:
    train(fclat,autoenc=True, epochs=50000, lr=0.001, name='test')
    trainFtD(fclatFtD,autoenc=True, epochs=50000, lr=0.001, name='testFtD')
else:
    fclat.load_state_dict(torch.load('test'))
    fclatFtD.load_state_dict(torch.load('testFtD'))





hull=ConvexHull(U)

def proj2hull(z, equations):
    """
    Project `z` to the convex hull defined by the
    hyperplane equations of the facets

    Arguments
        z: array, shape (ndim,)
        equations: array shape (nfacets, ndim + 1)

    Returns
        x: array, shape (ndim,)
    """
    G = np.eye(len(z), dtype=float)
    a = np.array(z, dtype=float)
    C = np.array(-equations[:, :-1], dtype=float)
    b = np.array(equations[:, -1], dtype=float)
    x, f, xu, itr, lag, act = solve_qp(G, a, C.T, b, meq=0, factorized=True)
    
    return x


def proj(u):
    u=torch.tensor(proj2hull(u[0].detach(), hull.equations),dtype=torch.float32).unsqueeze(0)
    return u    




def contact(D,P,invrigi=10.):
    u0=(torch.tensor([[56.,13.,0.]])-u_mean)/u_scale

    f=torch.zeros(1,3)
    f[0,0]=P
    f=(f-f_mean)/f_scale
    #u=fclat.backward(f, u0)
    u=fclatFtD.backward(f)
    u=proj(u)
    if u[0,1]*u_scale[1]+u_mean[1]+D<52.1:
        #no contact
        return u*u_scale+u_mean, f*f_scale+f_mean
    else:
        #contact : u[:,1]*u_scale[1]+u_mean[1]+D should equal 52.1 if the plan is rigid
        #           otherwise k(D1-D0)=f_z => D1=-f_z/k+D0
        #           and u[:,1]*u_scale[1]+u_mean[1]+D1 should equal 52.1
        

        f_z=(f*f_scale+f_mean)[0,2]
        D1=max(min(-f_z*invrigi+D,D),14.)
        u[:,1]=(52.1-D1-u_mean[1])/u_scale[1]


        u=proj(0.5*u+0.5*u0)
        x0=u[:,[0,2]]
        #x0[0,0]=x0[0,0]/2   
        #x0[0,1]=x0[0,1]*    0

        
        
        xn=nn.Parameter(x0)
        optimizer=torch.optim.Adam([xn],lr=0.05)
        
        def criterion(x,f):
            u=torch.zeros(1,3)
            u[:,[0,2]]=x     
            
            

            f_z=(f*f_scale+f_mean)[0,2]
            D1=max(min(-f_z*invrigi+D,D),14.)
            u[:,1]=(52.1-D1-u_mean[1])/u_scale[1]

            
            
            #u[:,1]=(52.1-D-u_mean[1])/u_scale[1]
            f=fclat(u)

            return torch.sum((f[0,0]-(P-f_mean[0])/f_scale[0])**2)  
        
        tol=1.e-5
        nmax=100
        i=0

        while criterion(xn,f)>tol and i<nmax:
            
            

            i+=1
            optimizer.zero_grad()
            err=criterion(xn,f)
            err.backward()
            optimizer.step()

            u=torch.zeros(1,3)
            u[:,[0,2]]=xn   
            
            

            f_z=(f*f_scale+f_mean)[0,2]
            D1=max(min(-f_z*invrigi+D,D),14.)
            u[:,1]=(52.1-D1-u_mean[1])/u_scale[1]


            #u[:,1]=(52.-D-u_mean[1])/u_scale[1]
        
            u=proj(u)
            f=fclat(u)


            #xn[0,0].data.clamp((0.-u_mean[0])/u_scale[0],(56.-u_mean[0])/u_scale[0])
            #xn[0,1].data.clamp((-100.-u_mean[2])/u_scale[2],(0.-u_mean[2])/u_scale[2])
            xn.data=u[:,[0,2]]
            


        

        return u*u_scale+u_mean, f*f_scale+f_mean











