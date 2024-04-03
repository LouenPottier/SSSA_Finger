# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 13:48:49 2024

@author: LP263296
"""

import numpy as np
import matplotlib.pyplot as plt

def beam_from_endpoint(x,z,ry,plot=False):
    '''
    Use bezier approximation to return the
    coordinate of the median line given the endpoint 
    coordinates in the (x,z) plan + the rotation
    '''
    
    x1=0
    x2=56/4
    x3=x-56/4*np.cos(-ry/180*np.pi)
    x4=x
    
    z1=13
    z2=13
    z3=z-56/4*np.sin(-ry/180*np.pi)
    z4=z
    
    t=np.linspace(0,1,100)
    
    X=(1-t)**3*x1 + 3*t*(1-t)**2*x2 + 3*t**2*(1-t)*x3 + t**3*x4
    Z=(1-t)**3*z1 + 3*t*(1-t)**2*z2 + 3*t**2*(1-t)*z3 + t**3*z4

    if plot:
        plt.scatter([-z1, -z2, -z3, -z4],[-x1, -x2, -x3, -x4])
        plt.plot(-Z,-X)
    return X,Z

def beam_from_endpoint_shape(x,z,ry,plot=False):
    '''
    Use bezier approximation to return the
    coordinate of the median line given the endpoint 
    coordinates in the (x,z) plan + the rotation
    '''
    
    x1=0
    x2=56/4
    x3=x-56/4*np.cos(-ry/180*np.pi)
    x4=x
    
    z1=13
    z2=13
    z3=z-56/4*np.sin(-ry/180*np.pi)
    z4=z
    
    t=np.linspace(0,1,100)
    
    X=(1-t)**3*x1 + 3*t*(1-t)**2*x2 + 3*t**2*(1-t)*x3 + t**3*x4
    Z=(1-t)**3*z1 + 3*t*(1-t)**2*z2 + 3*t**2*(1-t)*z3 + t**3*z4


        
        
    scale=6.07
    
    Xsh0=[0.,0.,35,35,57,57,86,86,92,97,97,126,126,132,138,138,166,166,172,178,178,206,206,213,218,218,246,246,253,259,259,287,287,293,300,300,327,327,332,339,339]
    Zsh0=[0.,86,86,72,72,111,111,45,40,45,107,107,45,40,45,102,102,45,40,45,98,98,45,40,45,92,92,45,40,45,87,87,45,40,45,82,82,45,40,40,0]
    
    Xsh0=np.array(Xsh0)/scale
    Zsh0=-(np.array(Zsh0)/scale)
    
    
    dX=np.diff(X)
    dZ=np.diff(Z)
    #norm=np.sqrt(dX**2+dZ**2)
    theta=np.zeros_like(X)
    theta[1:]=np.arctan2(dZ,dX)
    
    X0=np.linspace(0., 56.,100)
    Z0=np.zeros(100)+13
    
    dist=np.repeat(X0, len(Xsh0)).reshape(100,len(Xsh0)) - np.repeat(Xsh0, len(X0)).reshape(len(Xsh0),100).transpose(1,0)
    index=np.argmin(np.abs(dist),0)
    
    
    alpha=theta[index]*(1-np.linspace(0,0.,len(Xsh0)))
    
    Xsh=X[index]-np.sin(alpha)*Zsh0
    Zsh=Z[index]+np.cos(alpha)*Zsh0
    
    
    if plot:
        plt.scatter([-z1, -z2, -z3, -z4],[-x1, -x2, -x3, -x4])
        plt.plot(-Z,-X)
        plt.plot(-Zsh,-Xsh) 
    
    
    return np.concatenate((np.flip(Xsh),X)),  np.concatenate((np.flip(Zsh),Z))
    
    
