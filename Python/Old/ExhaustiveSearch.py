#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 19:20:33 2021

@author: Elmo
"""
from itertools import permutations
import numpy as np
import matplotlib.pyplot as plt
#%%
NodePop = 10

class Node:
    
    def __init__(self):
        
        self.position = 100*np.random.rand(2)



global Nodes        

Nodes = [Node() for i in range(NodePop)]
#%%
def calcDesirabillity(): # Calculates the inverse of Euclidean distance between all pairs of nodes (i,j)
    desirability = np.zeros((NodePop,NodePop))
    for i in range(NodePop):
        for j in range(NodePop):
            if i==j:                # The case when a node communicate with itself
                desirability[i,j]=0
            else:
                desirability[i,j] = ((Nodes[i].position[0]-Nodes[j].position[0])**2 + (Nodes[i].position[1]-Nodes[j].position[1])**2)**(-0.5)
    return desirability

desirability = calcDesirabillity()

def calcCostObj(walk):  # Calculates the cost objective function for its completed walk (i.e. total distance travelled around graph)
        
    costObj = 0
    for i in range(NodePop-1):
        costObj += desirability[walk[i],walk[i+1]]**(-1) # We take inverse of desirability value to find distance
    return costObj

permutations = list(permutations(range( NodePop)))

costs=[]

for i in range(len(permutations)):
    costs.append(calcCostObj(permutations[i]))
    
#%%
plt.figure()
plt.hist(costs, bins=30)
plt.show()






  
    
    