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

# Plotting
def plotDestinations(trajectory, numAgents):
    trajectory = trajectory.detach().cpu().numpy()
    colors = plt.cm.get_cmap('hsv', numAgents+1)
    for i in range(numAgents):
        plt.plot(trajectory[0, 4 * numAgents + 2 * i], trajectory[0, 4 * numAgents + 2 * i + 1], color=colors(i), marker='x', markersize=20)
        
def plotFinalPosition(trajectory, numAgents, marker):
    trajectory = trajectory.detach().cpu().numpy()
    colors = plt.cm.get_cmap('hsv', numAgents+1)
    for i in range(numAgents):
        plt.plot(trajectory[-1, 2 * i], trajectory[-1, 2 * i + 1], color=colors(i), marker=marker, markersize=5)
        
     
def plotTrajectories(trajectory, numAgents, linestyle, description):
    trajectory = trajectory.detach().cpu().numpy()
    colors = plt.cm.get_cmap('hsv', numAgents+1)
    for i in range(numAgents):
            plt.plot(trajectory[:, 2 * i], trajectory[:, 2 * i + 1], color=colors(i), linewidth=2, linestyle=linestyle, label=description if i==0 else '')

def plotTrajectoriesClean(trajectory, numAgents, description):
    trajectory = trajectory.detach().cpu().numpy()

    colors = plt.cm.get_cmap('hsv', numAgents+1)
    alphas = torch.linspace(0.15, 0.9, steps=trajectory.shape[0]).numpy()

    for i in range(numAgents):
        x = trajectory[:, 2 * i]
        y = trajectory[:, 2 * i + 1]
        
        for j in range(len(x)):
            plt.scatter(x[j], y[j], color=colors(i), alpha=alphas[j], s=200)

def plotFrame(my_learned_trajectory, real_trajectory, nEpochs, epoch_save, maxNumSamples, numAgents):
    # Plotting
    plt.rcParams.update({'font.size': 20})

    plt.figtext(0.12, 0.9, "Total iterations: " + str(nEpochs)+ " - Current iteration: " + str(epoch_save) ,fontsize="15")
    plotTrajectories(real_trajectory, numAgents, 'dotted', "Real trajectory")
    plotFinalPosition(real_trajectory, numAgents, 'o')

    plotTrajectoriesClean(my_learned_trajectory, numAgents, "Compared model")
    plotFinalPosition(my_learned_trajectory, numAgents, 'p')
    
    plotDestinations(real_trajectory, numAgents)


    plt.xlabel('x $[$m$]$', fontsize=25)
    plt.ylabel('y $[$m$]$', fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

EPOCHS_PER_FRAME = 500
def updateFrame(frame, myLearnSystem, initial_state, real_trajectory, nEpochs, maxNumSamples, 
                numAgents, device, simulation_time, step_size, path_checkpoints):
    epoch_save = EPOCHS_PER_FRAME * (frame+1)
    print("Frame "+str(frame)+": epoch= "+str(epoch_save))
    try:
        myLearnSystem.load_state_dict(torch.load(path_checkpoints+"/epoch_"+str(epoch_save)+".pth", map_location=device))
        myLearnSystem.eval()
        my_learned_trajectory = myLearnSystem.forward(initial_state.unsqueeze(dim=0).to(device), simulation_time, step_size).squeeze(dim=1)
        
        plt.clf()
        plotFrame(my_learned_trajectory, real_trajectory, nEpochs[-1], epoch_save, maxNumSamples, numAgents)

    except FileNotFoundError:
        print("Error frame("+str(frame)+"): Save not found from epoch "+ str(epoch_save))
    