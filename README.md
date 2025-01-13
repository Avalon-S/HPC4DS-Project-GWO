# HPC4DS_Project_GWO

<div style="text-align: center;">
  <img src="asset/logo.jpg" alt="logo" style="width:40%;"/>
  <p><em>The logo of this project.</em></p>
</div>

<div align="center">
    <table>
        <tr>
            <td align="center">
                <a href="https://avalon-s.github.io/"><strong>Yuhang Jiang</strong></a><br>
                University of Trento (DISI)
            </td>
            <td align="center">
                <a href="https://github.com/sabimechanic"><strong>Ebuka Nwafornso</strong></a><br>
                University of Trento (DISI)
            </td>
            <td align="center">
                <a href="https://github.com/Munyaradzi-tech"><strong>Comfort Munyaradzi</strong></a><br>
                University of Trento (DISI)
            </td>
        </tr>
    </table>
</div>

---

## Table of Contents

1. [Introduction](#introduction)
2. [Motivation](#motivation)
3. [Contribution](#contribution)
4. [Overview of Algorithms Ideas and Implementation](#overview-of-algorithms-ideas-and-implementation)
5. [Experiment Platform Introduction](#experiment-platform-introduction)
6. [Usage](#usage)

---

## Introduction

This project is based on the **High Performance Computing for Data Science** course group project at the [University of Trento (UNITN)](https://www.unitn.it). The main theme is the parallelization of the Grey Wolf Optimizer (GWO). All instructions and codes in the project are designed for UNITN's [HPC cluster](https://servicedesk.unitn.it/sd/en/service/cluster-hpc?id=unitrento_v2_service_card&sys_id=7698d0fec35bfd104cbb7055df013118).

---

## Motivation

- The original Grey Wolf Optimizer, while simple and effective, heavily relies on the top three wolves (**α**, **β**, **δ**) and lacks additional information fusion, leading to **unsatisfactory convergence speed** in many cases.
- GWO’s dependence on the top three wolves means that every iteration requires collecting information about all wolves to update positions, creating **a significant sorting and communication overhead** for the master process. This often results in the parallel algorithm having similar execution times to the serial one (the main cost comes from sorting), wasting computational resources.

[Back to Table of Contents](#table-of-contents)

---

## Contribution

1. **Proposed the HGT-GWO algorithm**:
   - Introduced global historical best positions and individual trend guidance (similar to velocity vectors).
   - Outperformed GWO on 3 test functions.
2. **Proposed a master-worker island parallelization scheme**:
   - Subpopulations operate independently, with `sync_interval` controlling the frequency of global synchronization.
   - Most of the sorting work is done on the worker processes, reducing communication and sorting overhead, and improving parallel efficiency.
3. **Experimental validation**:
   - HGT-GWO was first implemented in Python in October 2024 and demonstrated comprehensive superiority over GWO on 15 test functions.
   - The parallelized code was implemented using C, MPI, and OpenMP.

[Back to Table of Contents](#table-of-contents)

---

## Overview of Algorithms Ideas and Implementation

| **Aspect**               | **GWO**                                                                                              | **HGT-GWO**                                                                                         |
|---------------------------|-----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| **Optimization Mechanism** | Follows the standard GWO, with updates based purely on the positions of `alpha`, `beta`, and `delta`. | Incorporates additional historical guidance and trend adjustment into the position update formula.  |
| **Position Update Formula** | \[ X = (X1 + X2 + X3) / 3 \], where \( X1, X2, X3 \) are guided by `alpha`, `beta`, and `delta`.      | \[ X = (1 - α_w - β_w) * X_GWO + α_w * X_history + β_w * X_trend \], including historical and trend components. |
| **History Guidance**      | Not utilized.                                                                                      | Utilizes `previous_best_pos` (global historical best position) to guide the search process.         |
| **Trend Adjustment**      | Not included.                                                                                      | Accounts for the movement trend of individuals (`previous_positions`) to adjust their trajectory.   |
| **Data Stored**           | Only the current positions and fitness of all individuals.                                         | Stores: 1) Global historical best position (`previous_best_pos`), 2) Previous generation positions (`previous_positions`). |
| **Exploration vs Exploitation** | Balances exploration and exploitation purely based on `alpha`, `beta`, and `delta`.                 | Enhances exploitation by leveraging historical information and trend data, improving convergence.   |
| **Convergence Speed**     | Moderate.                                                                                          | Faster due to history and trend-based refinement.                                                   |
| **Execution Speed**       | Moderate, as sorting is the most time-consuming operation.                                         | Faster overall, as faster convergence reduces the number of iterations despite more operations per iteration, reduces sorting time. |
| **Avoiding Local Optima** | May struggle in multi-modal functions; prone to premature convergence.                             | Better at escaping local optima by utilizing historical and trend information.                      |
| **Computational Overhead** | Lower, as it does not require additional historical data.                                          | Moderate, due to the need for storing and processing historical positions and trends.                 |
| **Application Scenarios** | Suitable for simpler or small-scale optimization problems.                                         | More suitable for complex, high-dimensional, and multi-modal optimization problems.                 |

---

- The larger the `sync_interval`, the stronger the independence of subpopulations, leading to faster runtime but slower convergence.
- When `sync_interval=1`, the algorithm reverts to the standard GWO or HGT-GWO.

| **Code Version**     | **Responsibilities of Master Process (rank 0)**                                                                                                    | **Responsibilities of Worker Processes (rank > 0)**                                                         |
|-----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| **Standard Parallelization**        | - Initialize the entire population (`population`) and distribute it to worker processes (Scatter).                                               | - Receive the distributed subpopulation (`local_pop`) and calculate the fitness of assigned individuals.    |
|                       | - Gather the subpopulation results from all processes (Gather) and combine them into the complete population.                                    | - Send the calculated subpopulation back to the master process (Gather).                                   |
|                       | - Sort the complete population, update `alpha`, `beta`, `delta`, and all individual positions.                                                  | - Wait for the updated subpopulation to be distributed, then receive it and begin the next round of fitness calculations. |
|                       | - Redistribute the updated population's subgroups to worker processes, completing the synchronization of a new generation (Scatter).             |                                                                                                             |
| **Master-worker Island Parallelization**         | - Initialize the entire population (`population`) and distribute it to worker processes (Scatter).                                               | - Receive the distributed subpopulation (`local_pop`) and calculate the fitness of assigned individuals.    |
|                       | - Gather the top 3 best individuals (`local_top3`) from each worker process (Gather) and combine them into global candidates (`global_top3`).    | - Select the top 3 best individuals from the subpopulation (`local_top3`) and send them to the master process (Gather). |
|                       | - Sort the global best individuals (`global_top3`) and determine the global `alpha`, `beta`, `delta` wolves.                                     | - Receive the global best individuals (`alpha`, `beta`, `delta`) broadcasted by the master process (Broadcast). |
|                       | - Broadcast the global top 3 individuals (`alpha`, `beta`, `delta`) to all worker processes (Broadcast) to ensure global consistency.            | - Use the broadcasted global best individuals to update the positions of the local population and start the next round of calculations. |

**Some faster methods**
- Standard Parallelization: The master process only sends `alpha`, `beta`, and `delta` information back to each process, but the largest sorting overhead still occurs in the master process.
- Master-worker Island Parallelization: Let each child process's local `alpha`, `beta`, and `delta` directly update its population individuals, which will converge faster (essentially independent GWO), but this will dilute the idea of "Master-worker" too much.

[Back to Table of Contents](#table-of-contents)

---

## Experiment Platform Introduction

This repository contains the codes, scripts, and experimental results for GWO (serial, MPI, MPI+OpenMP) and HGT-GWO (serial, MPI, MPI+OpenMP). 
The `experiments` folder contains:
- `Standard_Parallelization_serial_mpi` version
- `Standard_Parallelization_hybrid` version
- `Master-Worker_Island_Parallelization_serial_mpi` version
- `Master-Worker_Island_Parallelization_hybrid` version 
**The above four use quick sort.**
- `bubble_sort/Standard_Parallelization` version
- `bubble_sort/Master-worker_Island_Parallelization` version 
**Prove that execution time is mainly affected by the sorting algorithm.**  
The time complexity of **bubble sort** is \( O(n^2) \), and the time complexity of **quick sort** is \( O(n \log n) \) (close to \( O(n) \)). This is why when the population size is halved (\( n \to \frac{n}{2} \)), the speedup ratio of the parallel solution using bubble sort is approximately 4, while the speedup ratio of the parallel solution using quick sort is approximately 2.
And this is why letting rank 0 sort the entire population results in almost the same serial and parallel run times. **Most of the time is spent on sorting.**

Due to time constraints, the framework code for running serial, MPI, and MPI+OpenMP together has not been fully integrated (the core definitions differ between MPI and MPI+OpenMP, requiring additional code restructuring and bug testing). However, combining the scripts would complete the full testing framework.

### **Parameter Settings**

| Parameter          | Configuration                       |
| ------------------ | ----------------------------------- |
| Maximum Iterations | 500                                 |
| Population Size    | 1000                                |
| Test Functions     | Sphere, Schwefel 2.22, Schwefel 1.2 |
| Dimension          | 256, 512, 1024                      |
| Cores              | 1, 2, 4, 8, 16, 32, 64              |
| Threads            | 1, 2, 4                             |
| Sorting Algorithms | Bubble sort, quick sort             |

### **Experimental Conclusion**

[Back to Table of Contents](#table-of-contents)

---

## Usage

To avoid bugs caused by relative paths, this project uses absolute paths. Before using it, please rename the folder under `experiments`, `experiments/bubble_sort` you want to use to `Project`, and replace `yuhang.jiang` with your `<user_id>`.

```bash
cd Project
qsub src/scripts/0.cleanup.sh # Clean all experiment files (compiled files, logs, results)
qsub src/scripts/1.compile.sh # Compile the code
qsub src/scripts/2.run_experiments.sh # Run serial, MPI, MPI+OpenMP executables
qsub src/scripts/3.generate_scripts.sh # Generate scripts needed for 2.run_experiments.sh
qsub src/scripts/4.del_tasks.sh # Query and delete tasks in the cluster (replace <user_id>)

```
[Back to Table of Contents](#table-of-contents)
