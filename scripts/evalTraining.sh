#!/bin/bash
policy=$1
architecture=$2
nAttLayers=$3
nEpochs=$4
maxNumSamples=$5
numAgents=$6
interval_policy=$7
interval_parameter=$8
increment_policy=$9
increment_parameter=${10}


python3 EvalTraining.py --policy $policy --architecture $architecture --nAttLayers $nAttLayers \
    --nEpochs $nEpochs --numTrain 20000 --numTests 20000 --maxNumSamples $maxNumSamples --seed_train 42 \
    --seed_data 42 --numAgents $numAgents --numAgents_train 4 --interval_policy $interval_policy --interval_parameter $interval_parameter \
    --increment_policy $increment_policy --increment_parameter $increment_parameter

