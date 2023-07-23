# -*- coding: utf-8 -*-
"""
Pèndol doble matplotlib 
@author: Pau
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

dot_phi =0
dot_theta = 0
phi = pi/3

#Random
theta = 0
#Modes normals
#Mode normal 2
#theta = -(math.sqrt(2))*phi
#Mode normal 1
#theta = (math.sqrt(2))*phi

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
t = np.linspace(0,100, int(6e4))
sol = odeint(dSdt, y0=s, t=t, args = (m1,m2,g,L1,L2))
PHI = sol.T[0]
dPHI = sol.T[1]
THETA = sol.T[2]
dTHETA = sol.T[3]

x1 = L1*np.sin(PHI)
y1 = -L1*np.cos(PHI)
x2 = x1 + L2*np.sin(THETA)
y2 = y1 - L2*np.cos(THETA)


fig = plt.figure()
ax = fig.add_subplot(111, xlim=(-2, 2), ylim=(-2.25, 1))

plt.axis('off')

pendol1, = ax.plot([],[], '-o', color = '#ffffe4', markevery = [False, True])
traca1, = ax.plot([],[], '-', lw = 0.5, alpha = 0.75, color = '#ff9408')
pendol2, = ax.plot([],[], '-o', color = '#ffffe4')
traca2, = ax.plot([],[], '-', lw = 0.5, alpha = 0.75, color = '#ff9408')
def init():
    pendol1.set_data([], [])
    traca1.set_data([],[])
    pendol2.set_data([], [])
    traca2.set_data([],[])
    
    return traca1, traca2, pendol1, pendol2
def animate(i):
    traca1.set_data([x1[:20*i]], [y1[:20*i]])
    pendol1.set_data([0,x1[20*i]], [0,y1[20*i]])
    traca2.set_data([x2[:20*i]], [y2[:20*i]])
    pendol2.set_data([x1[20*i],x2[20*i]], [y1[20*i],y2[20*i]])
    
    return traca1, traca2, pendol1, pendol2

fig.patch.set_facecolor('black') 
ani = animation.FuncAnimation(fig, animate, np.arange(int(len(t)/20)), blit = True, init_func = init)
dpi = 200
writer = animation.writers['ffmpeg'](fps=30)
ani.save('random.mp4',writer=writer,dpi=dpi)







