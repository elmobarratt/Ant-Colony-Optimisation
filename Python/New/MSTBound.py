#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 17:10:41 2022

@author: Elmo
"""
import numpy as np
from itertools import permutations
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import curve_fit
import time
import pandas as pd
import heldkarp as hk




def prim(G):

    V = G.shape[0]

    selected = np.zeros(V)
    no_edge = 0

    selected[0] = True
    
    mst =[]
    cost=0
    
    while (no_edge < V - 1):
        minimum = np.inf
        x = 0
        y = 0
        for i in range(V):
            if selected[i]:
                for j in range(V):
                    if ((not selected[j]) and G[i][j]):  
                        if minimum > G[i][j]:
                            minimum = G[i][j]
                            x = i
                            y = j
        mst.append([x,y])
        cost+=G[x][y]
        
        selected[y] = True
        no_edge += 1
        
    
    return (mst,cost)

def TSPonRandUnitDisk(n):
    
    
    nodes = np.zeros((n,2))
    
    for i in range(n):
        
            theta = np.random.random(1)*2*np.pi
            radius = (np.random.random(1))**0.5
            nodes[i,0] = radius* np.cos(theta)
            nodes[i,1] = radius* np.sin(theta)
    
    distanceMatrix  = np.zeros((n,n))
    
    for i in range(n):
        for j in range(n):
            if i==j:                # The case when a node communicate with itself
                distanceMatrix[i,j]=0
            else:
                distanceMatrix[i,j] = ((nodes[i,0]-nodes[j,0])**2 + (nodes[i,1]-nodes[j,1])**2)**(0.5)

    minCost,minPath = hk.held_karp(distanceMatrix)
    minPath.append(0)
    
    (mst,mstCost) = prim(distanceMatrix)
    
    
    fig, ax = plt.subplots()
    fig.set_size_inches(6,6)

    circle = plt.Circle((0, 0), 1, color=(0.9,1,1,0.8))
    ax.add_patch(circle)

    ax.scatter(nodes[:,0],nodes[:,1])
    ax.plot([nodes[minPath[i],0] for i in range(n+1)],[nodes[minPath[i],1] for i in range(n+1)], alpha=0.5)
    
    for a in mst:
        ax.plot([nodes[a[0],0],nodes[a[1],0]],[nodes[a[0],1],nodes[a[1],1]], c="r",alpha=0.5)
    
    plt.show()
    
    
            

    return mstCost,minCost



#%%
# numTrials = 10000
# nodePopRange = range(3,17)
numTrials = 20000
nodePopRange = np.arange(3,17)



data = []
for n in nodePopRange:

    for i in range(1,numTrials+1):
        mstCost,minCost = TSPonRandUnitDisk(n)
        
        data.append((n,minCost,mstCost))
        
        if(i%100==0):
            print("done n= {} trial {}".format(n,i))
        
df = pd.DataFrame(data, columns = ["n","SHP","MST","R"])


df.to_csv("MSTComparison.csv")
