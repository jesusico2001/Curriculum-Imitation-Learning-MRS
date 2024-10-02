#!/bin/bash
policy=$1
numAgents=$2
numSamples=$3

python3 DatasetGenerator/Generator.py --policy $policy --numTrain 20000 --numVal 2000 --numTests 2000 --numSamples $numSamples  --seed 42 --numAgents $numAgents

