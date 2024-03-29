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