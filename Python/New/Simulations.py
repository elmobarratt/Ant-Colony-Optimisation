#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 14:34:51 2022

@author: Elmo
"""
import numpy as np
from AntColonyOpt import ACO
import time
import pandas as pd
from multiprocessing import Manager,Process,Pool
from os import urandom
import tqdm 


numNodesRange = np.arange(10,17,1,dtype="int_")
numAgentsRange = np.array([50,100,200],dtype="int_")
pheromoneEvapConstRange = np.array([0.05,0.1,0.15])
betaRange =np.array([0.5,1,1.5])
numTrials = 150


maxNumCycles=250


data=None

data = Manager().list()

dimension =[numNodesRange.size, numAgentsRange.size, pheromoneEvapConstRange.size, betaRange.size, numTrials]
numComputations = np.prod(dimension)

def multiprocessingFunc(inputTuple):
    
    n,S,rho,beta = inputTuple
    
    np.random.seed(int.from_bytes(urandom(4), byteorder='little'))
    
    A = ACO(n,maxNumCycles,S,rho,1,beta)
                    
    
    A.simulateFull(method = "entropy", comments=False)
    A.heldKarp()
    
    interrupted = False
    if (A.entropy > A.minimumEntropy+0.03) or (A.cycleNum >= maxNumCycles):
        interrupted = True
    
    # if A.cycleNum>=maxNumCycles:
    #     data.append((n,S,rho,beta,exhaustiveSearchTime,'DNF', int(A.cycleNum), A.minimumCost, A.calcEntropy()))
    # else:
    data.append((int(n),int(S),rho,beta,A.empiricalMinimumCost,A.minimumCost,A.entropy, int(A.cycleNum), interrupted))
    
    # print("done: n={}, S={}, rho={}, beta={}".format(n,S,rho,beta)) 

processes =[]
tasks =[]

if __name__ == "__main__":
    
    for n in numNodesRange:
        for S in numAgentsRange:
            for rho in pheromoneEvapConstRange:
                for beta in betaRange:
                    for m in range(numTrials):    
                        
                        tasks.append((n,S,rho,beta))
                        
                        # p = Process(target=multiprocessingFunc, args=(n,S,rho,beta))
                        # processes.append(p)
                        # p.start()
                        
                    
                        
    
    
    with Pool(4) as p:
      r = list(tqdm.tqdm(p.imap(multiprocessingFunc, tasks), total=len(tasks)))
    
    
    
             
                

data =list(data)

dataFrame = pd.DataFrame(data, columns = ['n', 'S', 'rho', 'beta', 'SHP', 'ACO_SHP', "Finishing Entropy", "Finishing Cycle #", "Algorithm Interrupted"])
# reducedDataFrame = dataFrame[(dataFrame["ACO Search Time"] != "DNF")]

dataFrame.to_csv("ACOvsEmpirical.csv")

# dataFrame.loc[:,['Exhaustive Search Time', 'ACO Search Time']]




