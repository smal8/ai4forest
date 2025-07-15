#!/bin/bash
gpunodes=( $(sinfo -N | grep -i gpu | grep -Ev 'micro|gcp') )
pids=()

for node in "${gpunodes[@]}"
do
    if [[ $node == *"htc"* ]]; then
        sbatch -w $node script.sbatch
        pids+=($!)
    fi
done

for pid in ${pids[*]}; do
    wait $pid
done