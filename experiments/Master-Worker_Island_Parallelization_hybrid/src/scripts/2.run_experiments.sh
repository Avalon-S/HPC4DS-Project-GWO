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
mkdir -p $log_dir/parallel

echo "Running parallel experiments..."
for dim in 256 512 1024; do
    for cores in 2 4 8 16 32 64; do
        for threads in 1 2 4 8; do
            mkdir -p "$log_dir/parallel/$dim/${cores}_${threads}"
            qsub -o "$log_dir/parallel/$dim/${cores}_${threads}/parallel.out" \
                 -e "$log_dir/parallel/$dim/${cores}_${threads}/parallel.err" \
                 "$scripts_dir/$dim/${cores}_cores_${threads}_threads.sh"
        done
    done
done

echo "All experiments completed!"
