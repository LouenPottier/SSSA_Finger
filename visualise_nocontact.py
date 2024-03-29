# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 16:08:35 2023

@author: LP263296
"""

import streamlit as st 
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import torch
from streamlit_plotly_events import plotly_events

from load_models import fclat , u_mean, u_scale, f_mean, f_scale



from utils import beam_from_endpoint


if 'h_fclat' not in st.session_state:
    u0=(torch.tensor([[56.,13.,0.]])-u_mean)/u_scale
    st.session_state['h_fclat'] =  fclat.enc_u(u0)


with st.sidebar:
    st.subheader('Pressure inside the beam')
    
    P = st.slider('P',0.,0.1,0.,0.005)
    #fy = st.slider('fy',-20000,20000,0,50) 
    

    fx=0
    fz=0
    
    #f=torch.tensor([[fx,fy]],dtype=torch.float32)/5000
    
    def apply(P,fx,fz):
        f=torch.zeros(1,3)
        f[0,0]=P
        f[0,1]=fx
        f[0,2]=fz
        f=(f-f_mean)/f_scale
        f_lat = fclat.enc_f(f)
        
        st.session_state['h_fclat'] = fclat.backward_lat(f_lat,st.session_state['h_fclat'],lr=0.01,nmax=1000)
        
        
        
    st.button('Apply load',on_click=apply,args=(P,fx,fz))
    
    
    
    st.subheader('Back to origin in latent space')

    
    def raz():
        u0=(torch.tensor([[56.,13.,0.]])-u_mean)/u_scale
        st.session_state['h_fclat'] =  fclat.enc_u(u0)


    
    st.button('Reset',on_click=raz)
    
    
    




##### PLOT FCLAT

st.subheader("Neural Network without energy structure")

u=fclat.dec_u(st.session_state['h_fclat'])
f_pred=fclat(u)*f_scale+f_mean

fx=f_pred[0,1].detach()
fz=f_pred[0,2].detach()

u=u*u_scale + u_mean
X, Z = beam_from_endpoint(float(u[0,0]),float(u[0,1]),float(u[0,2]))

fig2 = make_subplots(rows=1, cols=2,  subplot_titles=("Backward prediction", "Latent space visualization"))

#subplot1
fig2.add_trace(
    go.Scatter(x=-Z,y=-X,     name='',
),
    row=1, col=1,

)


print(fx,fz)

fig2.add_annotation(x =  -Z[-1]-fz,
                   y =  -X[-1]-fx,
                   text = "",
                   xref = "x",
                   yref = "y",
                   showarrow = True,
                   arrowcolor='red',
                   arrowhead = 3,
                   arrowsize = 2,
                   ax = -Z[-1],
                   ay = -X[-1],
                   axref="x",
                   ayref='y',) 




fig2.update_layout(xaxis_range=[-100,0],yaxis_range=[-100,0])




st.plotly_chart(fig2)




