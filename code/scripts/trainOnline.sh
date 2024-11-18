#!/bin/bash
policy=$1
architecture=$2
nAttLayers=$3
nEpochs=$4
maxNumSamples=$5
numAgents=$6

python3 TrainModel_Online.py --policy $policy --architecture $architecture --nAttLayers $nAttLayers \
    --nEpochs $nEpochs --numTrain 20000 --numVal 2000 --maxNumSamples $maxNumSamples --seed_train 50 \
    --seed_data 42 --numAgents $numAgents
