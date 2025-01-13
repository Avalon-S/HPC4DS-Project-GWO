#!/bin/bash
#set max execution time
#PBS -l walltime=0:20:00

#set the execution queue
#PBS -q short_cpuQ
module load mpich-3.2 # Load the required module

# Configuration
BASE_DIR="/home/yuhang.jiang/Project/src/scripts"  # Root directory where the build script is stored
DIMENSIONS=(256 512 1024)  # Different dimensions
CORES_LIST=(2 4 8 16 32 64)       # Numbers of cores 

# Serialization format
generate_serial_script() {
    local dim=$1
    local output_dir="$BASE_DIR/$dim"
    mkdir -p "$output_dir"

    cat <<EOF > "$output_dir/1_core.sh"
#!/bin/bash
#PBS -l walltime=5:00:00
#PBS -l select=1:ncpus=1:mem=2gb -l place=pack:excl
#PBS -q short_cpuQ
module load mpich-3.2 # Load the required module

# Path
project_root="/home/yuhang.jiang/Project"
bin_dir="\$project_root/bin"
log_dir="\$project_root/logs/serial/${dim}" 
mkdir -p \$log_dir

# Dimension
dim=${dim}
echo "Running serial version for dimension: \$dim..."
for func_id in F1 F2 F3; do
    mpirun.actual -n 1 "\$bin_dir/GWO_serial_\${dim}" \$func_id \$dim \\
        > "\$log_dir/\${func_id}_output.log" 2> "\$log_dir/\${func_id}_error.log"

    mpirun.actual -n 1 "\$bin_dir/HGT_GWO_serial_\${dim}" \$func_id \$dim \\
        > "\$log_dir/\${func_id}_HGT_output.log" 2> "\$log_dir/\${func_id}_HGT_error.log"
done
EOF
    echo "Generated: $output_dir/1_core.sh"
}

# Parallelization format
generate_parallel_script() {
    local dim=$1
    local cores=$2
    local output_dir="$BASE_DIR/$dim"
    mkdir -p "$output_dir"

    cat <<EOF > "$output_dir/${cores}_cores.sh"
#!/bin/bash
#PBS -l walltime=5:00:00
#PBS -l select=1:ncpus=${cores}:mem=2gb -l place=pack:excl
#PBS -q short_cpuQ
module load mpich-3.2 # Load the required module

# Path
project_root="/home/yuhang.jiang/Project"
bin_dir="\$project_root/bin"
log_dir="\$project_root/logs/parallel/${dim}/${cores}"  
mkdir -p \$log_dir

# Dimension and cores
dim=${dim}
cores=${cores}

echo "Running parallel (MPI) versions for dimension: \$dim with \$cores cores..."
for func_id in F1 F2 F3; do
    mpirun.actual -n ${cores} "\$bin_dir/GWO_parallel_\${dim}" \$func_id \$dim \$cores \\
        > "\$log_dir/MPI_\${func_id}_output.log" 2> "\$log_dir/MPI_\${func_id}_error.log"

    mpirun.actual -n ${cores} "\$bin_dir/HGT_GWO_parallel_\${dim}" \$func_id \$dim \$cores \\
        > "\$log_dir/MPI_\${func_id}_HGT_output.log" 2> "\$log_dir/MPI_\${func_id}_HGT_error.log"
done

echo "Completed parallel experiments for dimension: \$dim with \$cores cores."
EOF
    echo "Generated: $output_dir/${cores}_cores.sh"
}

# main function
main() {
    for dim in "${DIMENSIONS[@]}"; do
        # Generate serial script
        generate_serial_script "$dim"

        # Generate parallel scripts
        for cores in "${CORES_LIST[@]}"; do
            generate_parallel_script "$dim" "$cores"
        done
    done
}
main
