#!/bin/bash
#PBS -l walltime=0:10:00

#set the execution queue
#PBS -q short_cpuQ

# Get the task ID list of the current user and remove the .hpc-hea suffix
job_ids=$(qstat -u yuhang.jiang | awk 'NR>2 {print $1}' | sed 's/\..*//')

# Iterate over the task IDs and delete the tasks
if [ -z "$job_ids" ]; then
    echo "There are no tasks to delete."
else
    echo "The following tasks are being deleted:"
    echo "$job_ids"
    for job_id in $job_ids; do
        qdel $job_id
        echo "Deleted tasks:$job_id"
    done
fi