#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 15:34:16 2022

@author: Elmo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


df = pd.read_csv("/Users/Elmo/Desktop/Maths/Year 4/Project/CSVs/ACOvsEmpirical.csv")

numNodesRange = np.arange(10,17,1,dtype="int_")
numAgentsRange = np.array([50,100,200],dtype="int_")
pheromoneEvapConstRange = np.array([0.05,0.1,0.15])
betaRange =np.array([0.5,1,1.5])


proportion =[]


for n in numNodesRange:
    for S in numAgentsRange:
        for rho in pheromoneEvapConstRange:
            for beta in betaRange:
                
                df1 = df[(df["n"]==n) & (df["S"]==S) & (df["rho"]==rho) & (df["beta"]==beta)]
                interruptedProportion = len(df1[df1["Algorithm Interrupted"] == True])/len(df1)
                accurateProportion = len(df1[(df1["Algorithm Interrupted"] == False) & (df1["ACO_SHC"] - df1["SHC"] < 0.0001)])/len(df1)
                inaccurateProportion = len(df1[(df1["Algorithm Interrupted"] == False) & (df1["ACO_SHC"]-df1["SHC"] >= 0.0001)])/len(df1)
                proportion.append((n,S,rho,beta,accurateProportion,inaccurateProportion,interruptedProportion))
            

dataAccuracyProportion = pd.DataFrame(proportion,columns = ['n', 'S', 'rho', 'beta', "Accurate %", "Inaccurate %", "Interrupted %"])





#%%

df["Error"]=(df["ACO_SHP"]-df["SHP"])/df["SHP"]

df1 = df[(df["Algorithm Interrupted"] == False) & (df["Error"] >= 0.0001)]



np.around(100 * df1.groupby("n").mean()["Error"],2)
np.around(100*np.sqrt(df1.groupby("n").var()["Error"]),2)

np.around(100 * df1.groupby("S").mean()["Error"],2)
np.around(100*np.sqrt(df1.groupby("S").var()["Error"]),2)

np.around(100 * df1.groupby("rho").mean()["Error"],2)
np.around(100*np.sqrt(df1.groupby("rho").var()["Error"]),2)

np.around(100 * df1.groupby("beta").mean()["Error"],2)
np.around(100*np.sqrt(df1.groupby("beta").var()["Error"]),2)


#%%

fig, ax= plt.subplots()
fig.set_size_inches((6,4))

ax.set_ylabel("Frequency of Simulations")
ax.set_xlabel("Exact cost of SHC")

ax.hist(df[df["Algorithm Interrupted"]==False]["SHC"],bins =50, label = "Uniterrupted / Complete ACO Simulations", color="#41d439")
ax.hist(df[df["Algorithm Interrupted"]==True]["SHC"],bins =50, label = "Interrupted ACO Simulations", color ="#c91818")


ax.axvline(x=df[df["Algorithm Interrupted"]==False]["SHC"].mean(), ls ="--", c="#9effae")
ax.axvline(x=df[df["Algorithm Interrupted"]==True]["SHC"].mean(), ls="--", c = "#ff9e9e" )

ax.legend(loc ="upper left")

fig.savefig("Interrupted_SHC_hist_small_n.png", dpi =300)


#%%

pVals =[]
for n in numNodesRange:
    
    uninterruptednData = df[(df["Algorithm Interrupted"]==False) & (df["n"] == n)]["SHC"]
    interruptednData = df[(df["Algorithm Interrupted"]==True) & (df["n"] == n)]["SHC"]
    
    
    
    a =stats.kstest(uninterruptednData,interruptednData)
    pVals.append(a[1])
    



fig, ax= plt.subplots(nrows=1, ncols=7)
fig.set_size_inches((8,1.5))

height = 0

ax[0].set_ylabel("Frequency of Simulations")

for i in range(7):

    
    if i>0:
        ax[i].set_yticks([])
    ax[i].set_title(r"$n={}$".format(numNodesRange[i]))
    ax[i].set_xlabel(r"$p={}$".format(np.round(pVals[i],4)))
    
    n1, bins1, patches1 =  ax[i].hist(df[(df["Algorithm Interrupted"]==False) & (df["n"] == numNodesRange[i])]["SHC"],bins =30, label = "Uniterrupted / Complete ACO Simulations", color="#41d439")
    ax[i].hist(df[(df["Algorithm Interrupted"]==True) & (df["n"] == numNodesRange[i])]["SHC"],bins =30, label = "Interrupted ACO Simulations", color ="#c91818")
    
    height = max(height,max(n1))
    
    # ax[i].step(np.sort(df[(df["Algorithm Interrupted"]==False) & (df["n"] == numNodesRange[i])]["SHC"]), np.linspace(0,1,len(df[(df["Algorithm Interrupted"]==False) & (df["n"] == numNodesRange[i])]["SHC"])), color="#41d439" )
    # ax[i].step(np.sort(df[(df["Algorithm Interrupted"]==True) & (df["n"] == numNodesRange[i])]["SHC"]), np.linspace(0,1,len(df[(df["Algorithm Interrupted"]==True) & (df["n"] == numNodesRange[i])]["SHC"])), color ="#c91818")

for i in range(7):
    ax[i].set_ylim(0,height*1.05)    


plt.savefig("test.png",bbox_inches='tight', dpi =300)

# ax.axvline(x=df[df["Algorithm Interrupted"]==False]["SHC"].mean(), ls ="--", c="#9effae")
# ax.axvline(x=df[df["Algorithm Interrupted"]==True]["SHC"].mean(), ls="--", c = "#ff9e9e" )

# ax.legend(loc ="upper left")

#%%


        
        

