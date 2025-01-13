#!/bin/bash
#set max execution time
#PBS -l walltime=0:20:00

#set the execution queue
#PBS -q short_cpuQ
module load mpich-3.2 # Load the required module

# Configuration
BASE_DIR="/home/yuhang.jiang/Project/src/scripts"  # Root directory where the build script is stored
DIMENSIONS=(256 512 1024)  # Different dimensions
CORES_LIST=(2 4 8 16)       # Numbers of cores
THREADS_LIST=(1 2 4)            # Number of threads for OpenMP

# Parallelization format
generate_parallel_script() {
    local dim=$1
    local cores=$2
    local threads=$3
    local ncpus=$((cores * threads))  # Calculate total ncpus
    local output_dir="$BASE_DIR/$dim"
    mkdir -p "$output_dir"

    cat <<EOF > "$output_dir/${cores}_cores_${threads}_threads.sh"
#!/bin/bash
#PBS -l walltime=5:00:00
#PBS -l select=1:ncpus=${ncpus}:mem=2gb -l place=pack:excl
#PBS -q short_cpuQ
module load mpich-3.2 # Load the required module

# Path
project_root="/home/yuhang.jiang/Project"
bin_dir="\$project_root/bin"
log_dir="\$project_root/logs/parallel/${dim}/${cores}_${threads}"  
mkdir -p \$log_dir

# Dimension, cores, and threads
dim=${dim}
cores=${cores}
threads=${threads}

# Set OpenMP threads
export OMP_NUM_THREADS=\$threads  # Each MPI process uses \$threads threads

for func_id in F1 F2 F3; do
    mpiexec -np ${cores} "\$bin_dir/GWO_hybrid_\${dim}" \$func_id \$dim \$cores \
        > "\$log_dir/Hybrid_\${func_id}_output.log" 2> "\$log_dir/Hybrid_\${func_id}_error.log"

    mpiexec -np ${cores} "\$bin_dir/HGT_GWO_hybrid_\${dim}" \$func_id \$dim \$cores \
        > "\$log_dir/Hybrid_\${func_id}_output.log" 2> "\$log_dir/Hybrid_\${func_id}_error.log"
done

echo "Completed parallel experiments for dimension: \$dim with \$cores cores and \$threads threads."
EOF
    echo "Generated: $output_dir/${cores}_cores_${threads}_threads.sh"
}

# main function
main() {
    for dim in "${DIMENSIONS[@]}"; do
        for cores in "${CORES_LIST[@]}"; do
            for threads in "${THREADS_LIST[@]}"; do
                generate_parallel_script "$dim" "$cores" "$threads"
            done
        done
    done
}

main
