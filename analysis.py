#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 09:08:23 2022

@author: tjards
"""
import matplotlib as plt

fig, ax = plt.subplots()
ax.plot(t_all[4::],metrics_order_all[4::,1])

ax.set(xlabel='time (s)', ylabel='mean distance from target [m]',
       title='Distance from Target')
ax.grid()

#fig.savefig("test.png")
plt.show()