#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 10:10:31 2022

@author: Elmo
"""




import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import curve_fit
import pandas as pd
import os
from matplotlib import rc

rc("pdf", fonttype=3)
rc('font',**{'family':'serif'})
rc('text', usetex=True)


os.environ["PATH"] = "/opt/anaconda3/bin:/opt/anaconda3/condabin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/Library/TeX/texbin:/usr/local/MacGPG2/bin"



df = pd.read_csv("/Users/Elmo/Desktop/Maths/Year 4/Project/CSVs/ACO_sim_large_n.csv")
nodePopRange = np.array([20,25,30,35,40,45])


df["Delta"] = df["ACO_SHC"]-df["MST"]

# nSmalldf =  pd.read_csv("/Users/Elmo/Desktop/Maths/Year 4/Project/CSVs/MSTComparison.csv")
# nSmalldf["Delta"] = nSmalldf["SHC"]-nSmalldf["MST"]

# nSmallACO = pd.read_csv("/Users/Elmo/Desktop/Maths/Year 4/Project/CSVs/ACOvsEmpirical.csv")


#%%


# Percentage of 300 trials for a given n that were uninterrupted 

df1 = df[df["Algorithm Interrupted"]==False].groupby("n").size()/10



#%%
df2 = df[df["Algorithm Interrupted"] == False]


fig,ax =plt.subplots(len(nodePopRange),3)
fig.set_dpi(300)
fig.set_size_inches(6,5)

x1=np.linspace(np.floor(min(df2["MST"])),np.ceil(max(df2["MST"])),50)
x2=np.linspace(np.floor(min(df2["ACO_SHC"])),np.ceil(max(df2["ACO_SHC"])),50)
# x3=np.linspace(1,2,60)
x3=np.linspace(np.floor(min(df2["Delta"])),max(df2["Delta"]),40)

ax[0][0].set_title(r"$f(T^{\min}_n)$")
ax[0][1].set_title(r"$f(\mathbf{\tilde{w}}_n)$")
ax[0][2].set_title(r"$\tilde{\Delta}_n$")

mstHeight = 0
shcHeight = 0
deltaHeight = 0

for i in range(len(nodePopRange)):
    
    MST = df2[df2["n"]==nodePopRange[i]]["MST"]
    SHP = df2[df2["n"]==nodePopRange[i]]["ACO_SHC"]
    # R = df2[df2["n"]==nodePopRange[i]]["Delta"]
    R=df2[df2["n"]==nodePopRange[i]]["Delta"]
    

    
    
    ax[i][0].text(-1,0.25,r"$n={}$".format(nodePopRange[i]))

    n1, bins1, patches1 = ax[i][0].hist(MST, bins=x1, density = True, color = "#4dd0eb")
    n2, bins2, patches2 = ax[i][1].hist(SHP, bins=x2, density = True, color = "#77d4b1")
    n3, bins3, patches3 = ax[i][2].hist(R, bins=x3, density = True, color= "#b3f26f")
    
    ax[i][0].set_yticks([])
    ax[i][1].set_yticks([])
    ax[i][2].set_yticks([])
    
    ax[i][0].set_xticks([])
    ax[i][1].set_xticks([])
    ax[i][2].set_xticks([])
    
    mstHeight = max(mstHeight, max(n1))
    shcHeight = max(shcHeight, max(n2))
    deltaHeight = max(deltaHeight, max(n3))
    
for i in range(len(nodePopRange)):
    
    ax[i][0].set_ylim(0,mstHeight*1.05)
    ax[i][1].set_ylim(0,shcHeight*1.05)
    ax[i][2].set_ylim(0,deltaHeight*1.05)

    

ax[-1][0].set_xticks(np.arange(np.floor(min(df2["MST"])),np.ceil(max(df2["MST"])),1))
ax[-1][1].set_xticks(np.arange(np.floor(min(df2["ACO_SHC"])),np.ceil(max(df2["ACO_SHC"])),1))
# ax[-1][2].set_xticks([1,2])
ax[-1][2].set_xticks(np.arange(np.floor(min(df2["Delta"])),max(df2["Delta"]),0.5))

plt.savefig("large_n_Histogram_MST_SHC_Delta.png",bbox_inches='tight', dpi =300)


#%%
def estFunc1(x,a,b,c):
    return a* (x-b)**0.5 + c


k1,k2,k3 = 1.1023647 , 2.47805211, 1.92260287

def estFunc2(x,a,b,c,d):
    return a* np.sqrt(x) + b *x **(c) +d

k4,k5,k6,k7 = 0.45820196, -1.10867801,  0.34189248,  2.28701057


conf=0.1

SHC_means = np.array(df2[["n","ACO_SHC"]].groupby("n").mean()).T[0]
SHC_quantiles = np.array([df2[["n","ACO_SHC"]].groupby("n").quantile(conf/2),df2[["n","ACO_SHC"]].groupby("n").quantile(1-conf/2)]).T[0]
SHC_variances = np.array(df2[["n","ACO_SHC"]].groupby("n").var()).T[0]

Delta_means = np.array(df2[["n","Delta"]].groupby("n").mean()).T[0]
Delta_quantiles = np.array([df2[["n","Delta"]].groupby("n").quantile(conf/2),df2[["n","Delta"]].groupby("n").quantile(1-conf/2)]).T[0]
Delta_variances = np.array(df2[["n","Delta"]].groupby("n").var()).T[0]



fig2,ax2= plt.subplots(2)
fig2.set_dpi(300)
fig2.set_size_inches(6,5)

xRange = np.arange(15,51,1)


ax2[0].scatter(nodePopRange, SHC_means, c="r", label=r"Sample mean of $f(\mathbf{\tilde{w}}_n)$")
ax2[0].fill_between(nodePopRange, SHC_quantiles[:,0], SHC_quantiles[:,1], color='r', alpha=0.1, label="90\% CI of sample")
ax2[0].plot(xRange,estFunc1(xRange,k1,k2,k3),c="k",label = r"$\alpha(n) = k_1\sqrt{n-k_2} +k_3$", lw=1)

ax2[0].set_ylim(0, np.ceil(max(df2["ACO_SHC"])))
ax2[0].set_xlim(10,55)
ax2[0].legend(loc= 'lower right')
# ax2[0].set_xlabel(r"Number of nodes, $n$")
ax2[0].set_ylabel("Cost")

ax2[1].scatter(nodePopRange, Delta_means, c="r", label=r"Sample mean of $\tilde{\Delta}_n$")
ax2[1].fill_between(nodePopRange, Delta_quantiles[:,0], Delta_quantiles[:,1], color='r', alpha=0.1, label="90\% CI of sample")
ax2[1].plot(xRange,estFunc2(xRange,k4,k5,k6,k7),c="k",label = r"$\beta(n) = k_4\sqrt{n} +k_5 n^{k_6} +k_7$", lw=1)

ax2[1].set_ylim(0, 2.5)
ax2[1].set_xlim(10,55)
ax2[1].legend(loc = 'lower right')
ax2[1].set_xlabel(r"Number of nodes, $n$")
ax2[1].set_ylabel("Cost")

plt.savefig("Predictive_Model_vs_ACO_Data.png",bbox_inches='tight', dpi =300)
#%%
nRange = np.arange(20,46,5)

PercentageErrorInSHC = 100*(df2[["n","ACO_SHC"]].groupby("n").mean()["ACO_SHC"] - estFunc1(nRange,k1,k2,k3))/df2[["n","ACO_SHC"]].groupby("n").mean()["ACO_SHC"]
PercentageErrorInDelta = 100*(df2[["n","Delta"]].groupby("n").mean()["Delta"] - estFunc2(nRange,k4,k5,k6,k7))/df2[["n","Delta"]].groupby("n").mean()["Delta"]


#%%
nRange = np.arange(20,46,5)

ErrorInSHC = df2[["n","ACO_SHC"]].groupby("n").mean()["ACO_SHC"] - estFunc1(nRange,k1,k2,k3)
ErrorInDelta = df2[["n","Delta"]].groupby("n").mean()["Delta"] - estFunc2(nRange,k4,k5,k6,k7)


