from cmath import sqrt
import matplotlib.pyplot as plt
from matplotlib import colors
import similaritymeasures

import torch

def obtainInitialState(trajectory):
    return trajectory[0]

# Euclidian square mean
def L2_loss(u, v):
    return (u - v).pow(2).mean()

def area_loss(t1, t2):
    pass

# Smoothness
def getSmoothness(trajectory, numAgents, step_size):
    ax, ay = getAccelerations(trajectory, numAgents, step_size)
   
    return  (torch.sum(ax**2) + torch.sum(ay**2) ) / numAgents

def getAccelerations(trajectory, numAgents, step_size):
    # Extract speeds
    vx = trajectory[:, 2*numAgents:4*numAgents:2]
    vy = trajectory[:, 2*numAgents+1:4*numAgents:2]
    
    # Aceleracionrd crentrales con diferencias finas
    ax_central = (vx[2:, :] - vx[:-2, :]) / (2 * step_size)
    ay_central = (vy[2:, :] - vy[:-2, :]) / (2 * step_size)
    
    # Aceleraciones en los extremos con diferencias hacia adelante/atrás
    ax_start = (vx[1] - vx[0]) / step_size
    ax_end = (vx[-1] - vx[-2]) / step_size
    ay_start = (vy[1] - vy[0]) / step_size
    ay_end = (vy[-1] - vy[-2]) / step_size
    
    # Combinar todos los valores de aceleración
    ax = torch.cat((ax_start.unsqueeze(0), ax_central, ax_end.unsqueeze(0)))
    ay = torch.cat((ay_start.unsqueeze(0), ay_central, ay_end.unsqueeze(0)))
    return ax, ay
# Distances between robots       
def getDistances(trajectory, numAgents):
    positions = trajectory[:, :2*numAgents].reshape(-1, numAgents, 2)

    x = positions[:,:, 0]
    x1 = torch.kron(x, torch.ones((1,numAgents), device=trajectory.device ))
    x2 = x.repeat(1,numAgents)

    y = positions[:,:, 1]
    y1 = torch.kron(y, torch.ones((1,numAgents), device=trajectory.device ))
    y2 = y.repeat(1,numAgents)

    x_diff = abs(x1-x2).reshape(-1,numAgents,numAgents)
    y_diff = abs(y1-y2).reshape(-1,numAgents,numAgents)

    return torch.sqrt(pow(x_diff, 2) + pow(y_diff, 2))

def avgAgentDist(trajectory, numAgents):
    dist = getDistances(trajectory, numAgents)

    avgDist_instant = (torch.sum(dist,(1,2)) / (pow(numAgents,2)-numAgents))
    avgDist_general = torch.sum(avgDist_instant,0) / avgDist_instant.size()[0]

    return avgDist_general

def minAgentDist(trajectory, numAgents):
    dist = getDistances(trajectory, numAgents)
    dist[:, range(numAgents), range(numAgents)] = float('inf')

    minDist_instant, _ = torch.min(dist,2)
    minDist_instant, _ = torch.min(minDist_instant,1)

    minDist_general, _ = torch.min(minDist_instant,0)
    
    return minDist_general

def getAreaBetweenCurves(curve1, curve2, numAgents):
    positions1 = curve1.reshape(-1, numAgents, 8)[:,:,:2]
    positions2 = curve2.reshape(-1, numAgents, 8)[:,:,:2]

    total = 0.0
    for i in range(numAgents):
        total += similaritymeasures.area_between_two_curves(positions1[:,i,:], positions2[:,i,:])
    return total / numAgents
