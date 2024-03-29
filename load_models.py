# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 11:24:35 2023

@author: LP263296
"""

import torch
from models import Model, ModelFtD, FCNN, LEBNN


u_mean=torch.tensor([ 48.8330,  30.3924, -58.2864])
u_scale=torch.tensor([ 6.4992,  6.4992, 26.2561]) #differ from std, x and z use same scale
f_mean=torch.tensor([ 0.0500, -0.0450,  0.3152])
f_scale=torch.tensor([0.0318, 0.2662, 0.2662]) # differ from std, fx and fz use same scale


hidden_size=16
latent_size=3


fclat= Model(FCNN(latent_size,hidden_size,latent_size)   ) 

"""fclat= Model(LEBNN(latent_size,2*hidden_size,latent_size),
             enc_u=FCNN(3,hidden_size,latent_size),
             dec_u=FCNN(latent_size,hidden_size,3),
             enc_f=FCNN(3,hidden_size,latent_size),
             dec_f=FCNN(latent_size,hidden_size,3))"""

""",
             enc_u=FCNN(u.size(-1),hidden_size,latent_size),
             dec_u=FCNN(latent_size,hidden_size,u.size(-1)),
             enc_f=FCNN(f.size(-1),hidden_size,latent_size),
             dec_f=FCNN(latent_size,hidden_size,f.size(-1)))"""
             
             

fclat.load_state_dict(torch.load('test'))
