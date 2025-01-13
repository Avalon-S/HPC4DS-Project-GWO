#!/bin/bash
#set max execution time
#PBS -l walltime=0:20:00

#set the execution queue
#PBS -q short_cpuQ
module load mpich-3.2 # Load the required module

echo "Running: Compiling all GWO versions for all dimensions..."

project_root="/home/yuhang.jiang/Project"

# Compilation output directory
output_dir="$project_root/bin"
log_dir="$project_root/logs/compile" 
mkdir -p $output_dir
mkdir -p $log_dir

# Compile the serial version
for dim in 256 512 1024; do
    mpicc -std=c99 -o "$output_dir/GWO_serial_${dim}" \
        "$project_root/src/GWO/GWO_serial.c" \
        "$project_root/src/common/common_functions.c" \
        "$project_root/src/common/test_functions.c" -lm \
        > "$log_dir/GWO_serial_${dim}.out" 2> "$log_dir/GWO_serial_${dim}.err"
    echo "Compiled binary for dim=${dim} at: $output_dir/GWO_serial_${dim}"

    # Compile HGT-GWO serial version
    mpicc -std=c99 -o "$output_dir/HGT_GWO_serial_${dim}" \
        "$project_root/src/HGT-GWO/HGT-GWO_serial.c" \
        "$project_root/src/common/common_functions.c" \
        "$project_root/src/common/test_functions.c" -lm \
        > "$log_dir/HGT_GWO_serial_${dim}.out" 2> "$log_dir/HGT_GWO_serial_${dim}.err"
    echo "Compiled binary for HGT-GWO dim=${dim} at: $output_dir/HGT_GWO_serial_${dim}"
done

# Compile MPI version
for dim in 256 512 1024; do
    # Compile GWO parallel version
    mpicc -std=c99 -o "$output_dir/GWO_parallel_${dim}" \
        "$project_root/src/GWO/GWO_parallel.c" \
        "$project_root/src/common/common_functions.c" \
        "$project_root/src/common/test_functions.c" -lm \
        > "$log_dir/GWO_parallel_${dim}.out" 2> "$log_dir/GWO_parallel_${dim}.err"
    echo "Compiled MPI binary for dim=${dim} at: $output_dir/GWO_parallel_${dim}"

    # Compile HGT-GWO parallel version
    mpicc -std=c99 -o "$output_dir/HGT_GWO_parallel_${dim}" \
        "$project_root/src/HGT-GWO/HGT-GWO_parallel.c" \
        "$project_root/src/common/common_functions.c" \
        "$project_root/src/common/test_functions.c" -lm \
        > "$log_dir/HGT_GWO_parallel_${dim}.out" 2> "$log_dir/HGT_GWO_parallel_${dim}.err"
    echo "Compiled HGT-GWO MPI binary for dim=${dim} at: $output_dir/HGT_GWO_parallel_${dim}"
done

echo "All versions compiled successfully and stored in $output_dir."
