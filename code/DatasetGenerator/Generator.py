import argparse
import torch
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")
from DatasetGenerator import RealSystemBuilder

def generateDataset(name, realSys, na, numData, numSamples, time, step_size):
    simulation_time = torch.linspace(0, time-step_size, int(time/step_size))
    num_trajectories  = int(numData/time*step_size*numSamples)

    demonstrations = torch.zeros(numSamples, numData, 8 * na)
    print('Generating ' +name+ ' dataset...')
    for k in range(num_trajectories):
        q_agents, p_agents               = realSys.generate_agents(na)
        q_dynamic, p_dynamic             = realSys.generate_leader(na)
        input                            = torch.cat((q_agents, p_agents, q_dynamic, p_dynamic))
        trajectory                       = realSys.sample(input, simulation_time, step_size)
        l                                = int(time/step_size/numSamples)

        # Avoid strange examples due to numerical or initial configuration issues
        while torch.isnan(trajectory).any():
            q_agents, p_agents               = realSys.generate_agents(na)
            q_dynamic, p_dynamic             = realSys.generate_leader(na)
            input                            = torch.cat((q_agents, p_agents, q_dynamic, p_dynamic))
            trajectory                       = realSys.sample(input, simulation_time, step_size)
            l                                = int(time/step_size/numSamples)
        
        demonstrations[:, l*k:l*(k+1), :] = (trajectory.reshape(-1, numSamples, 8 * na).transpose(0, 1))
        print('\tInstance '+str(k)+'.')

    return demonstrations

def main(policy="FS", numTrain=20000, numVal=20000, numTests=20000, numSamples=5, seed=42, numAgents=4):

    # Set seed
    torch.manual_seed(seed)

    # Hyperparameters
    step_size       = 0.04
    time            = 10.0

    # Parameters
    parameters = RealSystemBuilder.buildParameters(policy, numAgents)
    print("Parameters built...\n")

    # Initialize the system to learn
    realSys = RealSystemBuilder.buildRealSystem(policy, parameters)
    print("Real System built... - Type= ", type(realSys), "\n")

    # Build datasets
    train_data = generateDataset("train", realSys, numAgents, numTrain, numSamples, time, step_size)
    val_data = generateDataset("val", realSys, numAgents, numVal, numSamples, time, step_size)
    test_data = generateDataset("test", realSys, numAgents, numTests, numSamples, time, step_size)
    
    # Store data
    torch.save(train_data, 'saves/datasets/'+policy+str(numAgents)+'_trainData_'+str(numTrain)+'_'+str(numSamples)+'_'+str(seed)+'.pth')
    torch.save(val_data, 'saves/datasets/'+policy+str(numAgents)+'_valData_'+str(numVal)+'_'+str(numSamples)+'_'+str(seed)+'.pth')
    torch.save(test_data, 'saves/datasets/'+policy+str(numAgents)+'_testData_'+str(numTests)+'_'+str(numSamples)+'_'+str(seed)+'.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset generator for LEMURS')
    parser.add_argument('--policy', type=str, nargs=1, help='control policy name')
    parser.add_argument('--numTrain', type=int, nargs=1, help='number of instances for training')
    parser.add_argument('--numVal', type=int, nargs=1, help='number of instances for validation')
    parser.add_argument('--numTests', type=int, nargs=1, help='number of instances for testing')
    parser.add_argument('--numSamples', type=int, nargs=1, help='number of samples per instance')
    parser.add_argument('--seed', type=int, nargs=1, help='seed to generate reproducible data')
    parser.add_argument('--numAgents', type=int, nargs=1, help='number of agents')
    args = parser.parse_args()
    main(args.policy[0], args.numTrain[0], args.numVal[0], args.numTests[0], args.numSamples[0], args.seed[0], args.numAgents[0])
