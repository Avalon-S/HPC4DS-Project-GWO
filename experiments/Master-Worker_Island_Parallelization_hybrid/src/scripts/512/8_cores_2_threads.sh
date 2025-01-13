#!/bin/bash
#PBS -l walltime=5:00:00
#PBS -l select=1:ncpus=16:mem=2gb -l place=pack:excl
#PBS -q short_cpuQ
module load mpich-3.2 # Load the required module

# Path
project_root="/home/yuhang.jiang/Project"
bin_dir="$project_root/bin"
log_dir="$project_root/logs/parallel/512/8_2"  
mkdir -p $log_dir

# Dimension, cores, and threads
dim=512
cores=8
threads=2

# Set OpenMP threads
export OMP_NUM_THREADS=$threads  # Each MPI process uses $threads threads

for func_id in F1 F2 F3; do
    mpiexec -np 8 "$bin_dir/GWO_hybrid_${dim}" $func_id $dim $cores         > "$log_dir/Hybrid_${func_id}_output.log" 2> "$log_dir/Hybrid_${func_id}_error.log"

    mpiexec -np 8 "$bin_dir/HGT_GWO_hybrid_${dim}" $func_id $dim $cores         > "$log_dir/Hybrid_${func_id}_output.log" 2> "$log_dir/Hybrid_${func_id}_error.log"
done

echo "Completed parallel experiments for dimension: $dim with $cores cores and $threads threads."
