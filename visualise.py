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

#from load_models import fclat 

from main import contact,  u_mean, u_scale, f_mean, f_scale, U, F, dP13, p, fclat


from utils import beam_from_endpoint


if 'h_fclat' not in st.session_state:
    u0=(torch.tensor([[56.,13.,0.]])-u_mean)/u_scale
    st.session_state['h_fclat'] =  fclat.enc_u(u0)


with st.sidebar:
    st.subheader('Pressure inside the beam')
    
    P = st.slider('P',0.,0.1,0.,0.01)
    D = st.slider('D',14.,24.,23.5,0.1)
    invrigi = st.slider('k^-1',0.,20.,0.,1.)

    


    #f=torch.tensor([[fx,fy]],dtype=torch.float32)/5000
    
    def apply(D,P,invrigi):
        u,f=contact(D,P,invrigi)
        u=(u-u_mean)/u_scale
        st.session_state['h_fclat'] = fclat.enc_u(u)
        
    st.button('Apply load',on_click=apply,args=(D,P,invrigi))
    
    
    
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

D1=max(min(-fz*invrigi+D,D),14.)

u=u*u_scale + u_mean
X, Z = beam_from_endpoint(float(u[0,0]),float(u[0,1]),float(u[0,2]))

fig2 = make_subplots(rows=1, cols=1,  subplot_titles=("Backward prediction"))

#subplot1
fig2.add_trace(
    go.Scatter(x=-Z,y=-X,     name='',
),
    row=1, col=1,

)



fig2.add_trace(
    go.Scatter(x=[D1-52.1,D1-52.1],y=[0,-56.],     name='',
),
    row=1, col=1,

)


fig2.add_trace(
    go.Scatter(x=[D-52.1,D-52.1],y=[0,-56.],     name='',
),
    row=1, col=1,

)


fig2.add_annotation(x =  -Z[-1]-fz*50,
                   y =  -X[-1]-fx*50,
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




fig2.update_layout(xaxis_range=[-56,0],yaxis_range=[-56,0])




st.plotly_chart(fig2)



##### PLOT BDD

index = torch.min(torch.abs(dP13-D)+torch.abs(p-P),0)[1]

st.subheader("Closest in DB")

u=U[[int(index)]]*u_scale+u_mean

f_pred=F[[int(index)]]*f_scale+f_mean

fx=f_pred[0,1].detach()
fz=f_pred[0,2].detach()

X, Z = beam_from_endpoint(float(u[0,0]),float(u[0,1]),float(u[0,2]))

fig3 = make_subplots(rows=1, cols=1,  subplot_titles=("Backward prediction"))

#subplot1
fig3.add_trace(
    go.Scatter(x=-Z,y=-X,     name='',
),
    row=1, col=1,

)

fig3.add_trace(
    go.Scatter(x=[D-52.1,D-52.1],y=[0,-56.],     name='',
),
    row=1, col=1,

)


fig3.add_annotation(x =  -Z[-1]-fz*50,
                   y =  -X[-1]-fx*50,
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




fig3.update_layout(xaxis_range=[-56,0],yaxis_range=[-56,0])




st.plotly_chart(fig3)



