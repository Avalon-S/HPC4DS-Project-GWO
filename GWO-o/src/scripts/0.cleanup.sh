#!/bin/bash
#PBS -l walltime=0:10:00

#set the execution queue
#PBS -q short_cpuQ

project_root="/home/yuhang.jiang/Project"
folders=("bin" "data" "logs")  # The folders to be cleaned.

echo "Cleaning up all *.sh.* files in $project_root..."


find "$project_root" -type f -name "*.sh.*" -exec rm -f {} \;

echo "Cleanup completed! All *.sh.* files in $project_root have been removed."

echo "Cleaning up files in the following folders under $project_root: ${folders[*]}"


for folder in "${folders[@]}"; do
    target_dir="$project_root/$folder"
    if [ -d "$target_dir" ]; then
        echo "Cleaning folder: $target_dir"
        # Delete file
        find "$target_dir" -type f -exec rm -f {} \;
        # Delete folder
        find "$target_dir" -mindepth 1 -type d -exec rm -rf {} \;
    else
        echo "Folder $target_dir does not exist, skipping."
    fi
done
echo "Cleanup completed! All files in the specified folders have been removed."
