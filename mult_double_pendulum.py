# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 23:04:10 2023

@author: 34626
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
import math 
from scipy.integrate import odeint
from math import pi

def sin(theta):
  return np.sin(theta)
def cos(theta):
  return np.cos(theta)

g = 9.81
L1 =1
L2=1
m1 = 1
m2 = 1

    #Modes normals
    #Mode normal 2
    #theta = -(math.sqrt(2))*phi
    #Mode normal 1
    #theta = (math.sqrt(2))*phi
    
t = np.linspace(0,100, int(6e4))  
def pendols(phi, theta):
    global t
    dot_phi =0
    dot_theta = 0


    def D2phi(theta, phi, Dphi, Dtheta):
      return (-g*(2*m1+m2)*sin(phi)-m2*g*sin(phi-2*theta)-2*sin(phi-theta)*m2*(Dtheta**2*L2+Dphi**2*L1*cos(phi-theta)))/(L1*(2*m1+m2-m2*cos(2*phi-2*theta)))
    def D2theta(theta, phi, Dphi, Dtheta):
      return (2*sin(phi-theta)*(Dphi**2*L1*(m1+m2)+g*(m1+m2)*cos(phi)+Dtheta**2*L2*m2*cos(phi-theta)))/(L2*(2*m1+m2-m2*cos(2*(phi-theta))))
    
    def dSdt(S,t, m1,m2,g,L1, L2):
        phi = S[0]
        Dphi = S[1]
        theta = S[2]
        Dtheta = S[3]
        d2phi = (-g*(2*m1+m2)*sin(phi)-m2*g*sin(phi-2*theta)-2*sin(phi-theta)*m2*(Dtheta**2*L2+Dphi**2*L1*cos(phi-theta)))/(L1*(2*m1+m2-m2*cos(2*phi-2*theta)))
        d2theta = (2*sin(phi-theta)*(Dphi**2*L1*(m1+m2)+g*(m1+m2)*cos(phi)+Dtheta**2*L2*m2*cos(phi-theta)))/(L2*(2*m1+m2-m2*cos(2*(phi-theta))))
        return np.array([Dphi, d2phi, Dtheta, d2theta])
    
    s= np.array([phi, dot_phi,theta, dot_theta])
    
    sol = odeint(dSdt, y0=s, t=t, args = (m1,m2,g,L1,L2))
    PHI = sol.T[0]
    dPHI = sol.T[1]
    THETA = sol.T[2]
    dTHETA = sol.T[3]
    
    x1 = L1*np.sin(PHI)
    y1 = -L1*np.cos(PHI)
    x2 = x1 + L2*np.sin(THETA)
    y2 = y1 - L2*np.cos(THETA)
    return np.array([x1, y1, x2, y2])

fig = plt.figure()
ax = fig.add_subplot(111, xlim=(-2, 2), ylim=(-2.4, 1))

plt.axis('off')
colors =[]
dt = 0.01
N = 50
colors = ['crimson','red','darkorange','orange','yellow', 'greenyellow','lime','aquamarine', 'cyan','blue' ]
PENDOLS1 = [plt.plot([], [],  '-o', color = '#ffffe4', markevery = [False, True], lw = 0.4)[0] for _ in range(N)] #lines to animate
TRACES1 = [plt.plot([],[], '-', lw = 0.5, alpha = 0.75)[0] for i in range(N)]
PENDOLS2 = [plt.plot([], [],  '-o', color = '#ffffe4', markevery = [False, True], lw = 0.4)[0] for _ in range(N)] #lines to animate
TRACES2 = [plt.plot([],[], '-', lw = 0.5, alpha = 0.75)[0] for i in range(N)]
patches = PENDOLS1 + TRACES1 + PENDOLS2 + TRACES2
coord = [np.array(pendols(pi/3, pi/2+dt*j)) for j in range(N)]

def init():
    for pendol1, traca1, pendol2, traca2 in zip(PENDOLS1, TRACES1, PENDOLS2, TRACES2):
        pendol1.set_data([], [])
        traca1.set_data([],[])
        pendol2.set_data([], [])
        traca2.set_data([],[])
        
    return patches
def animate(i):
    for j, patch in enumerate(zip(PENDOLS1, TRACES1, PENDOLS2, TRACES2)):
        x1, y1, x2, y2 = coord[j]
        pendol1 = patch[0]
        traca1 = patch[1]
        pendol2 = patch[2]
        traca2 = patch[3]
        traca1.set_data([x1[:20*i]], [y1[:20*i]])
        pendol1.set_data([0,x1[20*i]], [0,y1[20*i]])
        if i>25:
            traca2.set_data([x2[int(20*i-5e2):20*i]], [y2[int(20*i-5e2):20*i]])
        else:
            traca2.set_data([x2[:20*i]], [y2[:20*i]])
        pendol2.set_data([x1[20*i],x2[20*i]], [y1[20*i],y2[20*i]])
    return  patches

fig.patch.set_facecolor('black') 
ani = animation.FuncAnimation(fig, animate, np.arange(int(len(t)/20)), blit = True, init_func = init)
dpi = 200
writer = animation.writers['ffmpeg'](fps=30)
ani.save('mult3.mp4',writer=writer,dpi=dpi)
