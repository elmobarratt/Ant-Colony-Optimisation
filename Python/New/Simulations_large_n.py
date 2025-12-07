#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 16:31:35 2022

@author: Elmo
"""

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


numNodesRange = np.array([20,25,30,35,40,45])

S= 150
rho=0.1
beta = 2

numTrials =400

maxNumCycles=300


data=None

data = Manager().list()


def multiprocessingFunc(n):
    
    
    np.random.seed(int.from_bytes(urandom(4), byteorder='little'))
    
    A = ACO(n,maxNumCycles,S,rho,1,beta)
                    
    
    A.simulateFull(method = "entropy", comments=False)
    A.calcMST()
    
    interrupted = False
    if (A.entropy > A.minimumEntropy+0.03) or (A.cycleNum >= maxNumCycles):
        interrupted = True
    

    data.append((int(n),A.minimumCost, A.mstCost ,A.entropy, int(A.cycleNum), interrupted))
    

processes =[]
tasks =[]

if __name__ == "__main__":
    
    

    for m in range(numTrials):   
        for n in numNodesRange:
                        
                        tasks.append(n)
                        
                        # p = Process(target=multiprocessingFunc, args=(n,S,rho,beta))
                        # processes.append(p)
                        # p.start()
                        
                    
                        
    
    
    with Pool(8) as p:
      r = list(tqdm.tqdm(p.imap(multiprocessingFunc, tasks), total=len(tasks)))
    
    
    
             
                

data =list(data)


df1 = pd.DataFrame(data, columns = ['n', 'ACO_SHC', 'MST', "Finishing Entropy", "Finishing Cycle #", "Algorithm Interrupted"])

df2 = pd.read_csv("/Users/Elmo/Desktop/Maths/Year 4/Project/CSVs/ACO_sim_large_n.csv")

dataFrame = pd.concat([df1,df2])

dataFrame.to_csv("ACO_sim_large_n_new.csv")





