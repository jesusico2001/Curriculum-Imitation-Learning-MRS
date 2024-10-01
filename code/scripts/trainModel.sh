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


python3 TrainModel.py --policy $policy --architecture $architecture --nAttLayers $nAttLayers \
    --nEpochs $nEpochs --numTrain 20000 --numTests 20000 --maxNumSamples $maxNumSamples --seed_train 42 \
    --seed_data 42 --numAgents $numAgents --interval_policy $interval_policy --interval_parameter $interval_parameter \
    --increment_policy $increment_policy --increment_parameter $increment_parameter

# python3 evaluation/loss_evo/compareLosses.py \
#     --trainLosses evaluation/loss_evo/"$policy$numAgents$architecture"_"$nAttLayers"_"$nEpochs"_20000_20000_"$maxNumSamples"_42_42_$interval_policy"_"$interval_parameter".0_"$increment_policy"_"$increment_parameter".0/trainLosses.pth" \
#     --descriptions ""