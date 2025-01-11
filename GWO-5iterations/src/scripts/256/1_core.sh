#!/bin/bash
#PBS -l walltime=5:00:00
#PBS -l select=1:ncpus=1:mem=2gb -l place=pack:excl
#PBS -q short_cpuQ
module load mpich-3.2 # Load the required module

# Path
project_root="/home/yuhang.jiang/Project"
bin_dir="$project_root/bin"
log_dir="$project_root/logs/serial/256" 
mkdir -p $log_dir

# Dimension
dim=256
echo "Running serial version for dimension: $dim..."
for func_id in F1 F2 F3; do
    "$bin_dir/GWO_serial_${dim}" $func_id $dim \
        > "$log_dir/${func_id}_output.log" 2> "$log_dir/${func_id}_error.log"

    "$bin_dir/HGT_GWO_serial_${dim}" $func_id $dim \
        > "$log_dir/${func_id}_HGT_output.log" 2> "$log_dir/${func_id}_HGT_error.log"
done
