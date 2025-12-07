#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 13:10:24 2021

@author: Elmo
"""
import numpy as np
from itertools import permutations



alpha = 1       # Parameter for controlling the influence of pheremone in probability updating rule
beta = 1        # Parameter for controlling the influence of desirability in probability updating rule
rho = 0.05       # Parameter in interval (0,1) referred to as evaporation of pheromone constant

NodePop = 8       # Total number of Nodes on the graph
AntPop = 300        # Total number of Agents in the colony

cycleNum = 0   # Keeps track of current cycle number starting from cycle 0 to agree with python list indexing



class Node:
    
    def __init__(self):
        
        self.position = 100*np.random.rand(2)
        
class Agent:
    
    def __init__(self):
        
        self.position = 0   # Keeps track of node it currently is at
        self.partialWalk =[0] # Keeps track of walk up until its current position, 0 is there as all agents start at node 0
    
    def calcTransitionProb(self,cycle): # Calculates transition probability for the agent at a specific part of their partial walk 
    
        phrmMatrix = PheremoneHistory[cycle] # Uses data from the pheromone matrix at given cycle 

        transProb = np.zeros((NodePop))
        for i in range(NodePop):
            if i in self.partialWalk:  # Ensures the agent does not revisit nodes in a given cycle
                transProb[i] = 0
            else:
                transProb[i] = (phrmMatrix[self.position,i]**alpha)*(desirability[self.position,i]**beta)
        return transProb/transProb.sum(axis=0)
    
    def move(self,cycle):   # Moves the agent from one node to the next according to transition probability
        
        probabilities = self.calcTransitionProb(cycle)
        choice = np.random.choice(list(range(NodePop)),1, True, probabilities)
        
        self.position = choice
        self.partialWalk.append(self.position)
    
    def isComplete(self):
        
        if len(self.partialWalk) == NodePop:
            self.walk = self.partialWalk +[0]
            return True
        else:
            return False
    
    def calcCostObj(self):  # Calculates the cost objective function for its completed walk (i.e. total distance travelled around graph)
        
        costObj = 0
        for i in range(NodePop):
            costObj += desirability[self.walk[i],self.walk[i+1]]**(-1) # We take inverse of desirability value to find distance
        return costObj
    
    

    def calcChangeInPhrmMatrix(self,cycle): # Calculates how much influence the given agents route will have on the next cycles pheromone matrix
        
        changeInPhrmMatrix = np.zeros((NodePop,NodePop))
        costObj = self.calcCostObj()
                                   
        if (costObj<=optimalCostHistory[cycle]):    # This condition causes the elitist strategy only increasing pheromone trails if the cost is lower than the current best
            for i in range(NodePop):
                changeInPhrmMatrix[self.walk[i],self.walk[i+1]] = costObj**(-1)
            return changeInPhrmMatrix
                
        else:       # If no improved route is found, influence of this agent on pheromone trail is removed, so the zero matrix is returned
            return changeInPhrmMatrix



global Nodes        
Nodes = [Node() for i in range(NodePop)]
global Agents   
Agents = [Agent() for i in range(AntPop)]

#%%


def initPhrmMatrix():   # Generates the first uniformly spread pheremonoe value matrix for all edges
    phrmMatrix = np.full((NodePop,NodePop),1/(NodePop*(NodePop-1))) # Since there are n(n-1) directed edges in the complete graph on n dimensions, we spread initial pheremone value uniformly

    for i in range(NodePop):
        phrmMatrix[i,i]=0
    return phrmMatrix
        
PheremoneHistory =[initPhrmMatrix()] 



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
    



def calcPhrmMatrix(agents,cycle):
    
    
    
    ChangeInPhrmMatrix = np.zeros((NodePop,NodePop))
    
    for a in agents:
        ChangeInPhrmMatrix = ChangeInPhrmMatrix + a.calcChangeInPhrmMatrix(cycle)
        
    cVal = ChangeInPhrmMatrix.sum().sum()
    
    
    
    
    if cVal>0:  # The case where at least one agent has traversed a path with objective cost at least as good as the best path so far
        nextPhrmMatrix = (1-rho) * PheremoneHistory[cycle] + rho * ChangeInPhrmMatrix / cVal  
    else:       # The case where no agent has traversed a path with objective cost at least as good as the best path so far
        nextPhrmMatrix = PheremoneHistory[cycle]
        
    
    
    return nextPhrmMatrix
    
 
#%%
    
def simulate():
    
    global cycleNum
    cycleNum =0
    
    global optimalCostHistory # Records optimal cost indexed by cycle number
    optimalCostHistory = [np.inf]
    global optimalWalkHistory # Records optimal walk indexed by cycle number, set initially
    optimalWalkHistory = []

    for m in range(50):
        Agents = [Agent() for i in range(AntPop)] # Resets Agents class information
        costs =[]
        for a in Agents:
            while a.isComplete()==False:
                a.move(cycleNum)
            costs.append(a.calcCostObj())
        
        
        
        if min(costs) > optimalCostHistory[cycleNum]:
            optimalCostHistory.append(optimalCostHistory[cycleNum])
            
        else:
            optimalCostHistory.append(min(costs))
        
        optimalWalkHistory.append(Agents[np.argmin(np.array(costs))].walk)
        
            
        PheremoneHistory.append(calcPhrmMatrix(Agents, cycleNum))
        
        print("cycle {} complete.".format(str(m)))
        # print(Agents[10].walk)
        cycleNum +=1
        
    optimalCostHistory = [float(optimalCostHistory[i]) for i in range(len(optimalCostHistory))]                                                 # Converts Optimal Cost history into better displayed list
    optimalWalkHistory = [[int(optimalWalkHistory[i][j]) for j in range(len(optimalWalkHistory[i]))] for i in range(len(optimalWalkHistory))]   # Converts Optimal Walk history into better displayed list
    
    
simulate()


#%%


def calcCostObj(walk):  # Calculates the cost objective function for its completed walk (i.e. total distance travelled around graph)
        
    costObj = 0
    for i in range(NodePop-1):
        costObj += desirability[walk[i],walk[i+1]]**(-1) # We take inverse of desirability value to find distance
    return costObj

permutations = list(permutations(range(NodePop)))

costs=[]

for i in range(len(permutations)):
    costs.append(calcCostObj(permutations[i]))
 




#%%
class Simulate():
    
    
    def __init__(self):
        
        self.cycleNum =0
    
        self.optimalCostHistory  = [np.inf] # Records optimal cost indexed by cycle number
    
        self.optimalWalkHistory = [] # Records optimal walk indexed by cycle number, set initially

    def run(self):
        
        for m in range(50):
            Agents = [Agent() for i in range(AntPop)] # Resets Agents class information
            costs =[]
            for a in Agents:
                while a.isComplete()==False:
                    a.move(self.cycleNum)
                costs.append(a.calcCostObj())
        
        
        
        if min(costs) > optimalCostHistory[self.cycleNum]:
            optimalCostHistory.append(optimalCostHistory[self.cycleNum])
            
        else:
            optimalCostHistory.append(min(costs))
        
        optimalWalkHistory.append(Agents[np.argmin(np.array(costs))].walk)
        
            
        PheremoneHistory.append(calcPhrmMatrix(Agents, self.cycleNum))
        
        print("cycle {} complete.".format(str(m)))
        self.cycleNum +=1
        
    optimalCostHistory = [float(optimalCostHistory[i]) for i in range(len(optimalCostHistory))]                                                 # Converts Optimal Cost history into better displayed list
    optimalWalkHistory = [[int(optimalWalkHistory[i][j]) for j in range(len(optimalWalkHistory[i]))] for i in range(len(optimalWalkHistory))]   # Converts Optimal Walk history into better displayed list
    

#%%

import matplotlib.pyplot as plt

plt.figure()

plt.scatter(np.array([Nodes[i].position[0] for i in range(NodePop)]),np.array([Nodes[i].position[1] for i in range(NodePop)]))
plt.plot([Nodes[optimalWalkHistory[-1][i]].position[0] for i in range(NodePop+1)],[Nodes[optimalWalkHistory[-1][i]].position[1] for i in range(NodePop+1)])

plt.show()

#%%


with open('optimalWalkHistory.txt', 'w') as file:
    file.write(str(optimalWalkHistory))
    
with open('nodePos.txt', 'w') as file:
    file.write(str([([Nodes[optimalWalkHistory[-1][i]].position[0] for i in range(NodePop)]),[Nodes[optimalWalkHistory[-1][i]].position[1] for i in range(NodePop)]]))




           