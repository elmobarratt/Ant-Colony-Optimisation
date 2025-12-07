#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 15:11:07 2021

@author: Elmo
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
from matplotlib.animation import FuncAnimation, FFMpegWriter
import heldkarp as hk

class ACO:
    def __init__(self, numNodes, maxCycleNum,  numAgents,  pheromoneEvapConst, alpha_, beta_):
        self.M = maxCycleNum
        self.S = numAgents
        self.n = numNodes
        self.rho = pheromoneEvapConst
        self.alpha = alpha_
        self.beta = beta_
        
        
        self.cycleNum = 1
        
        self.minimumEntropy = np.log(self.n)/np.log(self.n*(self.n-1))
        
        self.empiricalMinimumCost, self.empiricalMinimumPath = None,None
        self.mst, self.mstCost = None, None 
        
        self.history = None
        
        self.repEntropyCounter = 0
        
        self.nodes = np.zeros((self.n,2))
        for i in range(self.n):
            theta = np.random.random(1)*2*np.pi
            radius = (np.random.random(1))**0.5
            self.nodes[i,0] = radius* np.cos(theta)
            self.nodes[i,1] = radius* np.sin(theta)
        
        
        
        
        self.minimumCost = 2*self.n
        self.minimumPath = list(range(self.n))+[0]
        self.entropy = 1
        
        self.pheromoneMatrix = np.full((self.n,self.n),(1/(self.n*(self.n-1))))
        for i in range(self.n):
            self.pheromoneMatrix[i,i] =0
            
        self.desirabilityMatrix  = np.zeros((self.n,self.n))
        self.distanceMatrix = np.zeros((self.n,self.n))
        for i in range(self.n):
            for j in range(self.n):
                if i==j:                # The case when a node communicate with itself
                    self.desirabilityMatrix[i,j]=0
                else:
                    self.desirabilityMatrix[i,j] = ((self.nodes[i,0]-self.nodes[j,0])**2 + (self.nodes[i,1]-self.nodes[j,1])**2)**(-0.5)
                    self.distanceMatrix[i,j] = 1/self.desirabilityMatrix[i,j]
    
    def reset(self):
        self.cycleNum = 1
        self.history = None
        self.minimumCost = (self.n)
        self.minimumPath = list(range(self.n))+[0]
        self.pheromoneMatrix = np.full((self.n,self.n),(1/(self.n*(self.n-1))))
        for i in range(self.n):
            self.pheromoneMatrix[i,i] =0
    
    
    def calcEntropy(self):
        entropy = 0
        logConst = np.log(self.n*(self.n-1))
        for i in range(self.n):
            for j in range(self.n):
                if i!=j:
                    entropy -= self.pheromoneMatrix[i,j]*np.log(self.pheromoneMatrix[i,j])/logConst
        self.entropy = entropy
        return entropy   
        
       
    def exhaustiveSearch(self):
        
        if self.n>16:
            print("WARNING: Too many nodes to brute force")
        elif self.empiricalMinimumCost != None:
            print("Brute force search already executed, check variables ACO.empiricalMinimumCost, ACO.empiricalMinimumPath")
        else:
            perms = list(permutations(range(1,self.n)))
            perms = [(0,) + perms[i] + (0,) for i in range(len(perms))]
            
            BFminCost = self.n
            BFminPath = list(range(self.n+1))
            
            for p in perms:
                costObj = 0
                for i in range(self.n):
                    costObj += self.desirabilityMatrix[p[i],p[i+1]]**(-1)
                
                if costObj <= BFminCost:
                    BFminCost = costObj
                    BFminPath = list(p)
                
            (self.empiricalMinimumCost, self.empiricalMinimumPath) =(BFminCost,BFminPath)
            
    def heldKarp(self):
        if self.n>25:
             print("WARNING: Too many nodes to use Held-Karp algorithm")
        elif self.empiricalMinimumCost != None:
            print("Brute force search already executed, check variables ACO.empiricalMinimumCost, ACO.empiricalMinimumPath")
        else:
            
            a,b =hk.held_karp(self.distanceMatrix)
            (self.empiricalMinimumCost, self.empiricalMinimumPath)=a,b+[0]
    
    def simulateAgent(self):
        
        position = np.random.choice(list(range(self.n)))
        partialWalk = [position]
        
        for k in range(self.n-1):
            
            transitionProb = np.zeros(self.n)
            for i in range(self.n):
                if i in partialWalk:  # Ensures the agent does not revisit nodes in a given cycle
                    transitionProb[i] = 0
                else:
                    transitionProb[i] = (self.pheromoneMatrix[position,i]**self.alpha)*(self.desirabilityMatrix[position,i]**self.beta)
            transitionProb = transitionProb/transitionProb.sum(axis=0)
            
            position = np.random.choice(list(range(self.n)),1, True, transitionProb)[0]
            partialWalk.append(position)
        partialWalk.append(partialWalk[0])
        
        
        costObj = 0
        for i in range(self.n):
            costObj += self.desirabilityMatrix[partialWalk[i],partialWalk[i+1]]**(-1) # We take inverse of desirability value to find distance
        
        return (partialWalk, costObj)
    
    
    def simulateCycle(self):
        
        changeInPheromoneMatrix = np.zeros((self.n,self.n))
        
        cycleMinimumCost = self.minimumCost
        cycleMinimumPath = self.minimumPath
        
        
        for s in range(self.S):
            
            (agentPath,agentCost) = self.simulateAgent()
            
            if (agentCost<=self.minimumCost):  # This condition causes the elitist strategy only increasing pheromone trails if the cost is lower than the current best
                for i in range(self.n):
                    changeInPheromoneMatrix[agentPath[i],agentPath[i+1]] += agentCost**(-1)
                
                if agentCost<= cycleMinimumCost:
                    
                    cycleMinimumCost = agentCost
                    cycleMinimumPath = agentPath
        
        cVal = changeInPheromoneMatrix.sum().sum()
        
        if cVal>0:  # The case where at least one agent has traversed a path with objective cost at least as good as the best path so far
            self.minimumCost = cycleMinimumCost 
            self.minimumPath = cycleMinimumPath 
            self.pheromoneMatrix = (1-self.rho) * self.pheromoneMatrix + self.rho * changeInPheromoneMatrix / cVal  
        
        
        
        oldEntropy = self.entropy
        self.calcEntropy()
        
        if oldEntropy== self.entropy:
            self.repEntropyCounter +=1
        else:
            self.repEntropyCounter = 0
            
        self.cycleNum += 1


    def simulateFull(self, method = None, saveHistory = False, comments = True):
        
        if saveHistory:
            self.history  = ([self.minimumPath], [self.minimumCost], [self.pheromoneMatrix], [self.calcEntropy()])
        
        if method == "exhaustive" and self.n<14:
            
            
            if self.empiricalMinimumCost == None:
                self.exhaustiveSearch() 
            
        
            
        def algorithmRunning():
        
            if method == "entropy" and self.cycleNum <= self.M:
                if (self.entropy > self.minimumEntropy+0.03) and (self.repEntropyCounter < 50):
                    return True
                else:
                    return False
            
            elif method == "exhaustive" and self.cycleNum <= self.M:
                if self.n<14:
                    
                    if self.minimumCost > self.empiricalMinimumCost:
                        return True
                    else:
                        return False
                else:
                    print("ERROR: Too many nodes to brute force")
                    return False
            else:
                return False
           
         
            
        while algorithmRunning() == True:
            
            self.simulateCycle()
                
            if saveHistory:
                self.history[0].append(self.minimumPath)
                self.history[1].append(self.minimumCost)
                self.history[2].append(self.pheromoneMatrix)
                self.history[3].append(self.entropy)
            
            if comments:
                x= self.minimumPath[:-1]
                reorderedPath = [x[(i+x.index(0))%len(x)] for i in range(len(x))]
                reorderedPath.append(reorderedPath[0])
                
                print("Cycle {0:4}: Cost - {1:5}   Entropy - {2:5}      Path - {3} ".format(self.cycleNum,np.round(self.minimumCost,3),np.round(self.entropy,3),reorderedPath))
                        
    
    def kSearch(self,k):
        
        repMinimumPath = self.minimumPath[:self.n]
        
        for i in range(self.n):
            
            repMinimumPath = list(np.roll(repMinimumPath,1))
            
            
            perms = list(permutations(repMinimumPath[1:1+k]))
            perms = [(repMinimumPath[0],) + perms[h] + (repMinimumPath[k+1],) for h in range(len(perms))]
            
            minTestPath = None
            minTestCost = np.inf
            
            for p in perms:
                costObj = 0
                for j in range(len(p)-1):
                    costObj += self.desirabilityMatrix[p[j],p[j+1]]**(-1)
                
                if costObj <= minTestCost:
                    
                    minTestPath = list(p)
                    minTestCost = costObj
                
            repMinimumPath[0:k+2] = minTestPath
        
        self.minimumPath = repMinimumPath + [repMinimumPath[0]]
        
        costObj = 0
        for j in range(self.n):
            costObj += self.desirabilityMatrix[self.minimumPath[j],self.minimumPath[j+1]]**(-1)
        
        self.minimumCost = costObj
        

        
            
            
                
                
    
    
    def calcMST(self):
        
        if self.mst == None:
        
            self.distanceMatrix 
            selected = np.zeros(self.n)
            no_edge = 0
            selected[0] = True
    
            mst =[]
            cost=0
            
            while (no_edge < self.n - 1):
                minimum = np.inf
                x = 0
                y = 0
                for i in range(self.n):
                    if selected[i]:
                        for j in range(self.n):
                            if ((not selected[j]) and self.distanceMatrix[i][j]):  
                                if minimum > self.distanceMatrix[i][j]:
                                    minimum = self.distanceMatrix[i][j]
                                    x = i
                                    y = j
                mst.append([x,y])
                cost+=self.distanceMatrix[x][y]
                
                selected[y] = True
                no_edge += 1
                
            
            self.mst = mst
            self.mstCost = cost 
        
        else:
            print("MST already computed, check variables: 'self.mst' and 'self.mstCost' ")
            
     
        
    def plot(self, plotSHP=True, plotMST= True):
        
        
        fig, ax = plt.subplots()
        fig.set_size_inches(6,6)

        circle = plt.Circle((0, 0), 1, color=(0.9,1,1,0.8))
        ax.add_patch(circle)

        ax.scatter(self.nodes[:,0],self.nodes[:,1])
        
        if plotSHP:
            ax.plot([self.nodes[self.minimumPath[i],0] for i in range(self.n+1)],[self.nodes[self.minimumPath[i],1] for i in range(self.n+1)], c="k")
        
        if plotMST:
            
            if self.mst ==None:
                self.calcMST()
            
            for a in self.mst:
                ax.plot([self.nodes[a[0],0],self.nodes[a[1],0]],[self.nodes[a[0],1],self.nodes[a[1],1]], c="r", alpha=0.5)
            
            
        
        plt.show()
    
    
    def animate(self, save=False):
        
        if self.history != None:
        
            fig, ax = plt.subplots()
            fig.set_size_inches(6,6)
            
            
            
            
            def update(frame):
                ax.clear()
                
                circle = plt.Circle((0, 0), 1, color=(0.9,1,1,0.8))
                ax.add_patch(circle)
                
                ax.text(-0.5,0.5,"Cycle {}".format(frame))
            
                ax.scatter(self.nodes[:,0],self.nodes[:,1])
                
                adaptedPheromone = np.maximum(self.history[2][frame],self.history[2][frame].T)
                
                # alphaScale = 1/np.max(adaptedPheromone)
                alphaScale = self.n
                
                lines =()
                
                for i in range(1,self.n):
                    for j in range(0,i):
                       lines + (ax.plot([self.nodes[i,0],self.nodes[j,0]],[self.nodes[i,1],self.nodes[j,1]],alpha = alphaScale* adaptedPheromone[i,j], color = 'k'),)
                
                return lines
            
            anim = FuncAnimation(fig, update, frames =range(self.cycleNum), blit=True)
            
            plt.show()
            if save:
                writer = FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=600)
        
                anim.save("PheromoneAnimation.mp4",writer=writer)
                    
        else:
            print("No history of the algorithm has been recorded. Try running ACO.simulateFull(saveHistory = True).")
    
    
    def generatePheromoneGif(self, save=False):
        
        if self.history != None:
            fig = plt.figure()
            ax=plt.axes(xlim=(-0.5,self.n-0.5),ylim=(-0.5,self.n-0.5) )
        
            fig.set_size_inches(6,6)
            im = plt.imshow(self.history[2][0].T,interpolation="nearest")
            
            
            ax.set_ylabel("Node j")
            ax.set_xlabel("Node i")

            
        
            
            def update(frame):
              
                
                
                
                im.set_array(self.history[2][frame].T)

                title = ax.set_title("Cycle {}: Entropy = {}".format(frame+1,str(round(self.history[3][frame],3))))
                
                
                return [im,title]
            
            anim = FuncAnimation(fig, update, frames =range(self.cycleNum), blit=True)
            
            plt.show()
            if save:
                writer = FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=600)
        
                anim.save("PheromoneMatrixAnimation.mp4",writer=writer)
            

        else:
            print("No history of the algorithm has been recorded. Try running ACO.simulateFull(saveHistory = True).")
            
#%%

            

fig, ax = plt.subplots()
fig.set_size_inches(6,6)


circle = plt.Circle((0, 0), 1, color="k")
ax.add_patch(circle)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

ax.scatter(A.nodes[:,0],A.nodes[:,1], c ='#7affe4', alpha =0.2)

for i in range(1,A.n):
    for j in range(0,i):
       ax.plot([A.nodes[i,0],A.nodes[j,0]],[A.nodes[i,1],A.nodes[j,1]],alpha = abs(min(np.random.normal(0,0.05),1)), color = '#7affe4')
       
ax.plot([A.nodes[A.minimumPath[i],0] for i in range(A.n+1)],[A.nodes[A.minimumPath[i],1] for i in range(A.n+1)], c="#00ffe5", alpha= 0.9)


fig.savefig("K_50_contents_graphic",bbox_inches='tight', dpi =300)


