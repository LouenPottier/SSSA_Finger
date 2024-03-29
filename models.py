# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 14:43:16 2023

@author: LP263296
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


from rbf import RBF, gaussian, multiquadric, bump2


##### ENCODERS & DECODERS


def flatten(u):
    """
    flatten the displacement and remove the fixed node
    supports batched displacements
    """
    return u[...,1:,:].flatten(start_dim=-2) 


def unflatten(u_flat):
    """
    opposite of flatten(u)
    supports batched displacements
    """
    u_flat=F.pad(u_flat,(2,0))
    new_shape=list(u_flat.shape)[:-1]+[u_flat.size(-1)//2,2]
    return u_flat.reshape(new_shape)
    



class EncoderU(nn.Module):
    """
    Fully-connected neural network that encode the displacement field
    in a low dimmentional latent space
    """    
    def __init__(self,num_nodes,latent_size,hidden_size):
        super(EncoderU, self).__init__()
        self.enc=nn.Sequential(nn.Linear((num_nodes-1)*2, hidden_size),
                                     nn.ELU(),
                                     nn.Linear(hidden_size, hidden_size),
                                     nn.ELU(),
                                     nn.Linear(hidden_size,  latent_size))
        #self.enc=nn.Linear((num_nodes-1)*2, latent_size)
        
        self.num_nodes=num_nodes
    def forward(self, u):
        #flatten the displacement and remove the fixed node
        u=u[...,1:,:].flatten(start_dim=-2) 
        #u0=self.u0(1)[...,1:,:].flatten(start_dim=-2) 
        u_lat=self.enc(u)#-self.enc(u0)
        return u_lat
    
    def u0(self,batch_size=None):        
        device=next(self.parameters()).device
        if batch_size is None:
            u0=torch.zeros(self.num_nodes,2,device=device)
        else:
            u0=torch.zeros(batch_size,self.num_nodes,2,device=device)
        u0[...,:,0]=torch.linspace(0,1,10,device=device)
        return u0  
    def u_lat_0(self,batch_size=1):
        return self(self.u0(batch_size))
    

class EncoderF(nn.Module):
    """
    Trivial encoder for the loading field : keep only the non-zero components
    and unflatten
    """    
    def __init__(self,nonzero_nodes_index=[9]):
        super(EncoderF, self).__init__()
        self.nonzero_nodes_index=nonzero_nodes_index

    def forward(self, f):
        f_lat=f[...,self.nonzero_nodes_index,:].flatten(start_dim=-2) 
        return f_lat



class DecoderU(nn.Module):
    """
    Fully-connected neural network that decode the displacement low 
    dimmentionnal latent variable into the displacement space
    """    
    def __init__(self,num_nodes,latent_size,hidden_size):
        super(DecoderU, self).__init__()
        self.dec=nn.Sequential(nn.Linear(latent_size, hidden_size), 
                             nn.ELU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ELU(),
                             nn.Linear(hidden_size, (num_nodes-1)*2)) 
        #self.dec=nn.Linear(latent_size,(num_nodes-1)*2)

    def forward(self, u_lat):
        
        #decode and add the fixed node displacement
        u_flat=F.pad(self.dec(u_lat),(2,0)) 
        
        #supports batched and non-batched
        new_shape=list(u_flat.shape)[:-1]+[u_flat.size(-1)//2,2]
        
        #unflatten the displacement field
        u=u_flat.view(new_shape)
        return u



class DecoderF(nn.Module):
    """
    Trivial decoder for the loading field : add zeros and unflatten
    """    
    def __init__(self,nonzero_nodes_index=[9],num_nodes=10):
        super(DecoderF, self).__init__()
        self.nonzero_nodes_index=nonzero_nodes_index
        self.num_nodes=num_nodes
        self.lin=nn.Linear(2*len(nonzero_nodes_index), 2*len(nonzero_nodes_index),bias=False)

    def forward(self, f_lat):
        
        #supports batched and non-batched
        old_shape=list(f_lat.shape)[:-1]+[f_lat.size(-1)//2,2]
        new_shape=list(f_lat.shape)[:-1]+[self.num_nodes,2]
                
        f=torch.zeros(new_shape, device=f_lat.device)
        
        #unflatten the force field
        f[...,self.nonzero_nodes_index,:]=f_lat.view(old_shape)      
        return f







##### ENERGY


class V(torch.nn.Module):
    """
    Learnable potential energy using radial basis neural network

    """

    def __init__(self, latent_size, num_basis_function):
        super(V, self).__init__()
        self.rbf = RBF(latent_size, num_basis_function, multiquadric)
        self.lin1 = nn.Linear(num_basis_function, 1,bias=False)
        self.num_basis_function = num_basis_function
        self.latent_size = latent_size
        self.F=nn.ELU()
    def forward(self, h):
        return self.F(self.lin1(self.rbf(h)))+ 1/2*torch.sum(h**2,-1).unsqueeze(-1)
    def grad(self,h):
        gradV = torch.vmap(torch.func.jacrev(self.forward))(h)[:,0,0,:]
        return gradV
    def H(self,h):
        h=torch.vmap(torch.func.hessian(self.forward))(h)[:, 0,0,:,:]
        return h

    
class W(nn.Module):
    """
    Bilinear work function 
    """

    def __init__(self, latent_size):
        super(W, self).__init__()
        
    def forward(self, u, f):
        return torch.einsum('bi,bi->b',u,f).unsqueeze(-1)
    
    def grad(self,u,f):
        return f

    def H(self,u,f):
        return torch.zeros(u.size(0),u.size(1),u.size(1))




class VFC(torch.nn.Module):
    """
    Learnable potential energy using radial basis neural network

    """

    def __init__(self, latent_size, hidden_size):
        super(VFC, self).__init__()
        self.fcnn=nn.Sequential(nn.Linear(latent_size, hidden_size), 
                             nn.ELU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ELU(),
                             nn.Linear(hidden_size, 1),
                             nn.ELU()
                             )

        self.latent_size = latent_size
    def forward(self, h):
        return self.fcnn(h) + 1/2*torch.sum(h**2,-1).unsqueeze(-1)
    def grad(self,h):
        gradV = torch.vmap(torch.func.jacrev(self.forward))(h)[:,0,:]
        return gradV
    def H(self,h):
        h=torch.vmap(torch.func.hessian(self.forward))(h)[:, 0,:,:]
        return h




##### LATENT SPACE MODELS



class FCNN(nn.Module):
    """
    Fully-connected neural network 
        f_lat=fcnn(u_lat) 
        
    """    
    def __init__(self, input_size, hidden_size, output_size):
        super(FCNN, self).__init__()
        self.fcnn=nn.Sequential(nn.Linear(input_size, hidden_size), 
                             nn.ELU(),
                             nn.Linear(hidden_size, output_size),
                             )

        
    def forward(self, u_lat):
        return self.fcnn(u_lat)
    

    def adam(self,f_lat,u_lat_0,lr,nmax):
        """
        Solve the backward problem by gradient descent
        """
        u_lat=nn.Parameter(u_lat_0)
        optimizer=torch.optim.Adam([u_lat],lr=lr)
        
        for i in range(nmax):
            optimizer.zero_grad()
            err=torch.sum((f_lat-self(u_lat))**2)
            err.backward()
            optimizer.step()
        return u_lat
    def newton(self,f_lat,u_lat_0,lr,nmax,tol=1.e-3):
        u_lat=u_lat_0
        
        i=0
        while torch.sum((self(u_lat)-f_lat)**2)>tol and i<nmax:
            i+=1
            gradRN=torch.vmap(torch.func.jacrev(self.forward))(u_lat)[:,:,:]
            u_lat=u_lat-lr*torch.linalg.solve(gradRN,self(u_lat)-f_lat)
        return u_lat

    def backward(self,f_lat, u_lat_0, lr=0.01, nmax=1000):        
        return self.newton(f_lat, u_lat_0, lr, nmax)




class LEBNN(nn.Module):
    """  
    Neural network using a latent space in which a learnable energy is
    conserved to learn the displacemets - loading relation.
    
    The conserved energy is V(u_lat)-W(u_lat,f_lat)
    V is a radially unbounded function
    W is a bilinear function : W(u_lat, f_lat) = u_lat^t .B.f_lat
    
    
    non-batched inputs are not supported
    """    
    def __init__(self, latent_size, hidden_size, rbnn=True):
        super(LEBNN, self).__init__()
        
        if rbnn:
            self.V=V(latent_size,hidden_size) 
        else:
            self.V=VFC(latent_size,hidden_size) 
            
            
        self.W=W(latent_size)
        
    def forward(self,u_lat):        

        
        gradV=self.V.grad(u_lat)
        f_lat=gradV
        return f_lat
    
    def E(self,u_lat,f_lat):
        return self.V(u_lat)-self.W(u_lat,f_lat)
    
    def newton(self,f_lat,u_lat_0,nmax=100,lr=1.,tol=1.e-5,plot=False):
        u_lat=u_lat_0
        i=0
        while torch.sum((self.V.grad(u_lat)-self.W.grad(u_lat,f_lat))**2)>tol and i<nmax:
            i+=1
            u_lat=u_lat-torch.linalg.solve(self.V.H(u_lat),self.V.grad(u_lat)-self.W.grad(u_lat,f_lat))
        return u_lat

    def adam(self,f_lat,u_lat_0,lr=0.01,nmax=1000,tol=1.e-4):
        u_lat=nn.Parameter(u_lat_0)
        optimizer=torch.optim.Adam([u_lat],lr=lr)
        
        
        i=0
        while torch.sum((self(u_lat)-f_lat)**2)>tol and i<nmax:
            i+=1
            optimizer.zero_grad()
            ener=torch.sum(self.E(u_lat,f_lat))
            ener.backward()
            optimizer.step()

        return u_lat.detach()
    
    def backward(self,f_lat, u_lat_0, lr=0.01, nmax=100):
        return self.adam(f_lat, u_lat_0, lr, nmax)


##### MODEL



class Model(nn.Module):
    
    def __init__(self, latent_model, enc_u=nn.Identity(), dec_u=nn.Identity(), enc_f=nn.Identity(), dec_f=nn.Identity()):
        super(Model, self).__init__()
        
        self.latent_model=latent_model
        self.enc_u=enc_u
        self.dec_u=dec_u
        self.enc_f=enc_f
        self.dec_f=dec_f
    
    def forward(self, u):        
        u_lat=self.enc_u(u)
        f_lat=self.latent_model(u_lat)
        f=self.dec_f(f_lat)
        return f
    def autoenc_u(self, u):
        return self.dec_u(self.enc_u(u))
    
    def autoenc_f(self, f):
        return self.dec_f(self.enc_f(f))
    
    def backward(self, f, u0, lr=0.01, nmax=100):
        u_lat_0=self.enc_u(u0).detach()
        f_lat=self.enc_f(f).detach()
        
        u_lat=self.latent_model.backward(f_lat, u_lat_0, lr, nmax)
        return self.dec_u(u_lat)
    def backward_lat(self, f_lat, u_lat_0, lr=0.01, nmax=500):
        u_lat=self.latent_model.backward(f_lat, u_lat_0, lr, nmax)
        return u_lat
    
    


class ModelFtD(nn.Module):
    """  
    """    
    def __init__(self, latent_model, enc_u=nn.Identity(), dec_u=nn.Identity(), enc_f=nn.Identity(), dec_f=nn.Identity()):
        super(ModelFtD, self).__init__()
        
        self.latent_model=latent_model
        self.enc_u=enc_u
        self.dec_u=dec_u
        self.enc_f=enc_f
        self.dec_f=dec_f
    
    def backward(self, f,u0=None, lr=None, nmax=None):        
        f_lat=self.enc_f(f)  
        u_lat=self.latent_model(f_lat)
        u=self.dec_u(u_lat)
        return u

    def backward_lat(self, f_lat,u0=None, lr=None, nmax=None):        
        u_lat=self.latent_model(f_lat)
        return u_lat
    
    def forward(self, u, f0=None, lr=0.01, nmax=500):
        if f0 is None:
            f0=torch.zeros_like(u)
        f_lat_0=self.enc_f(f0).detach()
        u_lat=self.enc_u(u).detach()
        f_lat=self.latent_model.backward(u_lat, f_lat_0, lr, nmax)
        return self.dec_f(f_lat)
    
    def autoenc_u(self, u):
        return self.dec_u(self.enc_u(u))
    
    def autoenc_f(self, f):
        return self.dec_f(self.enc_f(f))
    
    
    
    
    
