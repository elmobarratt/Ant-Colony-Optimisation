#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 16:06:12 2022

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


df = pd.read_csv("/Users/Elmo/Desktop/Maths/Year 4/Project/CSVs/MSTComparison.csv")
df["Delta"] = df["SHC"] - df["MST"]
nodePopRange = np.arange(min(df["n"]),max(df["n"])+1)

#%%


fig,ax =plt.subplots(len(nodePopRange),3)
fig.set_dpi(300)
fig.set_size_inches(6,5)

x1=np.linspace(0,np.ceil(max(df["MST"])),70)
x2=np.linspace(0,np.ceil(max(df["SHC"])),70)
# x3=np.linspace(1,2,60)
x3=np.linspace(0,max(df["Delta"]),60)

ax[0][0].set_title(r"$f(T^{\min}_n)$")
ax[0][1].set_title(r"$f(\mathbf{w}_n^*)$")
ax[0][2].set_title(r"$\Delta_n$")

mstHeight = 0
shcHeight = 0
deltaHeight = 0

for i in range(len(nodePopRange)):
    
    MST = df[df["n"]==nodePopRange[i]]["MST"]
    SHP = df[df["n"]==nodePopRange[i]]["SHC"]
    # R = df[df["n"]==nodePopRange[i]]["Delta"]
    R=df[df["n"]==nodePopRange[i]]["Delta"]
    

    
    
    ax[i][0].text(-3,0.25,r"$n={}$".format(nodePopRange[i]))

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

    

ax[-1][0].set_xticks(np.arange(0,np.ceil(max(df["MST"]))+0.1,1))
ax[-1][1].set_xticks(np.arange(0,np.ceil(max(df["SHC"]))+0.1,1))
# ax[-1][2].set_xticks([1,2])
ax[-1][2].set_xticks(np.arange(0,max(df["Delta"])+0.1,1))

fig.savefig("/Users/Elmo/Desktop/Maths/Year 4/Project/Images/Histogram_MST_SHC_Delta.png", dpi=300)


#%%
k = int(np.ceil(len(nodePopRange)/3))


fig3,ax3 =plt.subplots(3,k)
fig3.set_size_inches((12,10))

MSTmax = max(df["MST"])
Rmax = max(df["Delta"])

xVals = np.linspace(0,MSTmax, 100)

for i in range(len(nodePopRange)):
    
    MST = df[df["n"]==nodePopRange[i]]["MST"]
    SHP = df[df["n"]==nodePopRange[i]]["Delta"]
    
    # scipy.stats.linregress()
    
    ax3[i//k][i%k].hist2d(MST,SHP, 100, np.array([[0,MSTmax],[0,Rmax]] ))
    
    ax3[i//k][i%k].plot(xVals,xVals, lw=0.5, c="w")
    
    
    # ax3[i].set_yticks([0,SHPmax])
    # ax3[i].set_xticks()
    





#%%
def estFunc(x,a,b,c):
    return a* (x-b)**0.5 + c

testVariable ="SHC"
conf=0.1


nodePopRange = np.arange(min(df["n"]),max(df["n"])+1)

means = np.array(df[["n",testVariable]].groupby("n").mean()).T[0]
quantiles = np.array([df[["n",testVariable]].groupby("n").quantile(conf/2),df[["n",testVariable]].groupby("n").quantile(1-conf/2)]).T[0]
variances = np.array(df[["n",testVariable]].groupby("n").var()).T[0]



a,cov = curve_fit(estFunc,np.array(df[df["n"]>1]["n"]),np.array(df[df["n"]>1][testVariable]))

estMean = estFunc(nodePopRange,a[0],a[1],a[2])


fig2,ax2 =plt.subplots()
fig2.set_size_inches((6,4))
ax2.set_ylim(0,7.5)
ax2.set_xlim(0,22)
ax2.set_xlabel(r"Number of nodes, $n$")
ax2.set_ylabel(r"Cost")



# ax2.errorbar(nodePopRange,means, means-quantiles[:,0],quantiles[:,1]-means,fmt='o',barsabove=True, label=r"$ f(\mathbf{w}^*_n)$ sample data with a 90\% confidence interval")

ax2.fill_between(nodePopRange, quantiles[:,0], quantiles[:,1], color='r', alpha=0.1, label="90\% CI of sample")
ax2.scatter(nodePopRange, means, c="r", label=r"Sample mean of $f(\mathbf{w}^*_n)$")
xRange = np.linspace(0,20,80)
ax2.plot(xRange,estFunc(xRange,a[0],a[1],a[2]), c="k",label = r"$\alpha(n) = k_1\sqrt{n-k_2} +k_3$", lw=1)

ax2.legend(loc=4)




returnData = pd.DataFrame(np.vstack((nodePopRange.T,means.T,estMean.T,quantiles.T,variances.T)).T, columns = ["n","Mean","Estimate of Mean" ,"Quantile-{}".format(conf/2),"Quantile-{}".format(1-conf/2),"Variance"])


fig2.savefig("/Users/Elmo/Desktop/Maths/Year 4/Project/Images/RegressionSHC.png", dpi=300)  

#%%


def estFunc(x,a,b,c,d):
    return a* np.sqrt(x) + b *x **(c) +d

testVariable ="Delta"
conf=0.1


nodePopRange = np.arange(min(df["n"]),max(df["n"])+1)

means = np.array(df[["n",testVariable]].groupby("n").mean()).T[0]
quantiles = np.array([df[["n",testVariable]].groupby("n").quantile(conf/2),df[["n",testVariable]].groupby("n").quantile(1-conf/2)]).T[0]
variances = np.array(df[["n",testVariable]].groupby("n").var()).T[0]



a,cov = curve_fit(estFunc,np.array(df[df["n"]>=7]["n"]),np.array(df[df["n"]>=7][testVariable]))

estMean = estFunc(nodePopRange,a[0],a[1],a[2],a[3])


fig2,ax2 =plt.subplots()
fig2.set_size_inches((6,4))
ax2.set_ylim(0,2.5)
ax2.set_xlim(0,22)
ax2.set_xlabel(r"Number of nodes, $n$")
ax2.set_ylabel(r"Cost")



# ax2.errorbar(nodePopRange,means, means-quantiles[:,0],quantiles[:,1]-means,fmt='o',barsabove=True, label=r"$ f(\mathbf{w}^*_n)$ sample data with a 90\% confidence interval")

ax2.fill_between(nodePopRange, quantiles[:,0], quantiles[:,1], color='r', alpha=0.1, label="90\% CI of sample")
ax2.scatter(nodePopRange, means, c="r", label=r"Sample mean of $\Delta_n$")
xRange = np.linspace(2,20,80)
ax2.plot(xRange,estFunc(xRange,a[0],a[1],a[2],a[3]), c="k",label = r"$\beta(n) = k_4\sqrt{n} +k_5 x^{k_6} +k_7$", lw=1)

ax2.legend(loc=4)




returnData = pd.DataFrame(np.vstack((nodePopRange.T,means.T,estMean.T,quantiles.T,variances.T)).T, columns = ["n","Mean","Estimate of Mean" ,"Quantile-{}".format(conf/2),"Quantile-{}".format(1-conf/2),"Variance"])

fig2.savefig("/Users/Elmo/Desktop/Maths/Year 4/Project/Images/RegressionDelta.png", dpi=300)  



#%%
# Plotting variance of SHC and Delta against n

fig4, ax4 = plt.subplots()
fig4.set_size_inches((6,4))

ax4.set_xlabel(r"Number of nodes, $n$")
ax4.set_ylabel("Variance")

ax4.set_ylim(0, max(df.groupby("n").var()["SHC"])+0.1)

ax4.plot(nodePopRange, df.groupby("n").var()["SHC"] , c= "#77d4b1", label= r"Var$(f(\mathbf{w}_n^*))$")
ax4.plot(nodePopRange, df.groupby("n").var()["Delta"], c= "#b3f26f", label= r"Var$(\Delta_n)$")

ax4.legend()

# fig4.savefig("/Users/Elmo/Desktop/Maths/Year 4/Project/Images/SHC_Delta_variance_comparison.png", dpi=300)

#%%



