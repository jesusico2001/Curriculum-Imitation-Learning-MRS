#!/bin/bash
policy=$1
architecture=$2
nAttLayers=$3
nEpochs=$4
maxNumSamples=$5
numAgents=$6

bash scripts/trainModel.sh $policy $architecture $nAttLayers $nEpochs $maxNumSamples $numAgents fixed 0 fixed 0