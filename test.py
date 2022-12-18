#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 19:14:59 2021

@author: tjards
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#%% 
i = 0
r = 5
phi = 0
x = r*np.cos(phi)
y = r*np.sin(phi)
z = 0

mySize = 100
store = np.zeros([3,mySize])

while(1):   
    
    store[0,i] = x
    store[1,i] = y
    store[2,i] = z
      
    # x2 = r*np.cos(phi)
    # z2 = z*np.cos(phi) + 0.5*r*np.sin(phi)*np.sin(phi) # this 0.5 is mathemtically correct but 1 works better (why!!!)
    # y2 = r*np.sin(phi)- z2*np.sin(phi)
    
    xo = r*np.cos(phi)
    yo = r*np.sin(phi)
    zo = 5
       
    x = r*np.cos(phi)
    y = r*np.sin(phi)*np.cos(phi) #- 0.5*zo*np.sin(phi)
    z = 0.5*r*np.sin(phi)*np.sin(phi) # this 0.5 is mathemtically correct but 1 works better (why!!!)


    
    i += 1 

    phi += 0.1
    phi =  np.mod(phi, 2*np.pi)
    
    
    store2=store[0:2,:]
    
    if i > mySize-1:
        break
    

#%%


fig, ax = plt.subplots()
ax.set_title('Trajectory in x-y plane')
ax.set_xlabel('x-direction')
ax.set_ylabel('y-direction')
ax.set_xlim(-5,5)
ax.set_ylim(-5,5)
ax.plot(store[0,:], store[1,:])
ax.plot(0,0,'or')

# fig, ax = plt.subplots()
# ax.set_title('Trajectory in y-z plane')
# ax.set_xlabel('y-direction')
# ax.set_ylabel('z-direction')
# ax.set_xlim(-5,5)
# ax.set_ylim(-5,5)
# ax.plot(store[1,:], store[2,:])
# ax.plot(0,0,'or')

fig, ax = plt.subplots()
ax.set_title('Trajectory in x-z plane')
ax.set_xlabel('x-direction')
ax.set_ylabel('z-direction')
ax.set_xlim(-6,6)
ax.set_ylim(-6,6)
ax.plot(store[0,:], store[2,:])
ax.plot(0,0,'or')

#%% error bars plot

y = metrics_order_all[2::,1]
error = metrics_order_all[2::,2]
x = t_all[2::]

# orange
#colour = '#CC4F1B'
#colour2 = '#FF9848'

# green
# colour = '#3F7F4C'
# colour2 = '#7EFF99'

# blue
colour ='#1B2ACC'
colour2 = '#089FFF'



plt.plot(x, y, 'k', color=colour)
plt.fill_between(x, y-error, y+error,
    alpha=0.3, edgecolor=colour, facecolor=colour2,
    linewidth=0)
plt.title('Distance from target over time')
plt.xlabel('time [s]')
plt.ylabel('Mean and variance [m]')



