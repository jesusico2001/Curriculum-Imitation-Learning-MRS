#!/bin/bash
policy=$1
numAgents=$2
numSamples=$3

pwd 
python3 DatasetGenerator/Generator.py --policy $policy --numTrain 20000 --numTests 20000 --numSamples $numSamples --seed 42 --numAgents $numAgents

