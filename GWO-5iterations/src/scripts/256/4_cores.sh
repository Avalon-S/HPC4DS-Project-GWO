#!/bin/bash
#PBS -l walltime=5:00:00
#PBS -l select=1:ncpus=4:mem=2gb -l place=pack:excl
#PBS -q short_cpuQ
module load mpich-3.2 # Load the required module

# Path
project_root="/home/yuhang.jiang/Project"
bin_dir="$project_root/bin"
log_dir="$project_root/logs/parallel/256/4"  
mkdir -p $log_dir

# Dimension and cores
dim=256
cores=4

echo "Running parallel (MPI) versions for dimension: $dim with $cores cores..."
for func_id in F1 F2 F3; do
    mpirun.actual -n 4 "$bin_dir/GWO_parallel_${dim}" $func_id $dim $cores \
        > "$log_dir/MPI_${func_id}_output.log" 2> "$log_dir/MPI_${func_id}_error.log"

    mpirun.actual -n 4 "$bin_dir/HGT_GWO_parallel_${dim}" $func_id $dim $cores \
        > "$log_dir/MPI_${func_id}_output.log" 2> "$log_dir/MPI_${func_id}_error.log"
done

# Set OpenMP threads
export OMP_NUM_THREADS=1  # Each MPI process uses 1 thread

for func_id in F1 F2 F3; do
    mpiexec -np 4 "$bin_dir/GWO_hybrid_${dim}" $func_id $dim $cores \
        > "$log_dir/Hybrid_${func_id}_output.log" 2> "$log_dir/Hybrid_${func_id}_error.log"

    mpiexec -np 4 "$bin_dir/HGT_GWO_hybrid_${dim}" $func_id $dim $cores \
        > "$log_dir/Hybrid_${func_id}_output.log" 2> "$log_dir/Hybrid_${func_id}_error.log"
done

echo "Completed parallel experiments for dimension: $dim with $cores cores."
