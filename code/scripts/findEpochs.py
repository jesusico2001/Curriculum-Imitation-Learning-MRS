import argparse
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")
from TrainingTools import *
from LearnSystem import LearnSystemBuilder

def main(maxNumSamples, interval_policy='fixed', interval_parameter=500, increment_policy='fixed', increment_parameter=1):
 
    interval_function = CuadraticFunction.ModelInterval(interval_policy, interval_parameter)
    increment_function = CuadraticFunction.ModelIncrement(increment_policy, increment_parameter)

    epoch = 0
    k = 5

    while k <= maxNumSamples:
        epoch += interval_function.compute(k)
        k += increment_function.compute(k)

    epoch += interval_function.compute(k)

    print("Interval:  "+interval_policy+" - "+str(interval_parameter))
    print("Increment: "+increment_policy+" - "+str(increment_parameter)+"\n")
    print("EPOCHS = "+str(epoch))
    print("=======================\n")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LEMURS training for the flocking problem')
    parser.add_argument('--maxNumSamples', type=int, nargs=1, help='maximum number of samples per instance (min is 5)')
    parser.add_argument('--interval_policy', type=str, nargs=1, help='either fixed, linear or modulated')
    parser.add_argument('--interval_parameter', type=float, nargs=1, help='parameter: c, b or maxValue according to the policy')
    parser.add_argument('--increment_policy', type=str, nargs=1, help='either fixed, linear or modulated')
    parser.add_argument('--increment_parameter', type=float, nargs=1, help='parameter: c, b or maxValue according to the policy')

    args = parser.parse_args()
    main(args.maxNumSamples[0], args.interval_policy[0], args.interval_parameter[0], args.increment_policy[0] ,args.increment_parameter[0])
