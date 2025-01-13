#!/bin/bash
#set max execution time
#PBS -l walltime=0:20:00

#set the execution queue
#PBS -q short_cpuQ
module load mpich-3.2 # Load the required module

rm -f *.sh.*

# Project root directory
project_root="/home/yuhang.jiang/Project"
scripts_dir="$project_root/src/scripts"
log_dir="$project_root/logs"  # Log Directory
bin_dir="$project_root/bin"  # Executable file directory
mkdir -p $log_dir/serial
mkdir -p $log_dir/parallel


echo "Running serial experiments..."
for dim in 256 512 1024; do
    echo "Submitting serial script for dimension: $dim"
    mkdir -p "$log_dir/serial/$dim"
    qsub -o "$log_dir/serial/$dim/1_core.out" -e "$log_dir/serial/$dim/1_core.err" "$scripts_dir/$dim/1_core.sh"
done


echo "Running parallel experiments..."
for dim in 256 512 1024; do
    for cores in 2 4 8 16 32 64; do
        mkdir -p "$log_dir/parallel/$dim/$cores"
        qsub -W depend=afterok:$compile_job_id -o "$log_dir/parallel/$dim/$cores/parallel.out" -e "$log_dir/parallel/$dim/$cores/parallel.err" \
            "$scripts_dir/$dim/${cores}_cores.sh"

    done
done

echo "All experiments completed!"
