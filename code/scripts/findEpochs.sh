maxNumSamples=$1
interval_policy=$2
interval_parameter=$3
increment_policy=$4
increment_parameter=$5

python3 scripts/findEpochs.py --maxNumSamples $maxNumSamples --interval_policy $interval_policy --interval_parameter $interval_parameter \
    --increment_policy $increment_policy --increment_parameter $increment_parameter