#include "../common/GWO.h"
#include <mpi.h>

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Parse command-line arguments
    if (argc < 4) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <test_function_name> <dimension> <num_cores>\n", argv[0]);
        }
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    char *test_function_name = argv[1];
    g_dimension = atoi(argv[2]);
    int num_cores = atoi(argv[3]);

    if (g_dimension <= 0 || num_cores <= 0 || num_cores != size) {
        if (rank == 0) {
            fprintf(stderr, "Error: Invalid dimension or core count.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    // e.g. g_pop_size = 200; g_max_iter = 500; or use defaults from GWO.h

    // Retrieve test function info
    TestFunctionInfo *info = get_test_function_info(test_function_name);
    if (!info) {
        if (rank == 0) {
            fprintf(stderr, "Error: Unknown test function: %s\n", test_function_name);
        }
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    // Each process uses a different seed (for reproducibility)
    srand(12345 + rank);

    // Define MPI_WOLF
    MPI_Datatype MPI_WOLF;
    {
        int block_lengths[2]      = {g_dimension, 1};
        MPI_Aint displacements[2] = {
            offsetof(Wolf, position),
            offsetof(Wolf, fitness)
        };
        MPI_Datatype types[2]     = {MPI_DOUBLE, MPI_DOUBLE};
        MPI_Type_create_struct(2, block_lengths, displacements, types, &MPI_WOLF);
        MPI_Type_commit(&MPI_WOLF);
    }

    // Each process handles a portion of population
    int local_size = g_pop_size / size + ((rank < (g_pop_size % size)) ? 1 : 0);
    Wolf *local_pop = (Wolf *)malloc(local_size * sizeof(Wolf));
    if (!local_pop) {
        perror("Error allocating memory for local population");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    Wolf alpha, beta, delta;
    Wolf *population = NULL;

    // Logging directories and files
    char core_info[32];
    snprintf(core_info, sizeof(core_info), "%d_cores", num_cores);

    char convergence_dir[256], performance_dir[256];
    char convergence_file[256], performance_file[256];

    // Only rank 0 handles log directory and files
    if (rank == 0) {
        generate_directory_path(convergence_dir, sizeof(convergence_dir),
                                "/home/yuhang.jiang/Project/data/convergence",
                                test_function_name, g_dimension, core_info);

        generate_directory_path(performance_dir, sizeof(performance_dir),
                                "/home/yuhang.jiang/Project/data/performance_logs",
                                test_function_name, g_dimension, core_info);

        create_directory_recursively(convergence_dir);
        create_directory_recursively(performance_dir);

        snprintf(convergence_file, sizeof(convergence_file),
                 "%s/convergence_GWO_parallel.txt", convergence_dir);
        snprintf(performance_file, sizeof(performance_file),
                 "%s/performance_log_GWO_parallel.txt", performance_dir);

        // Initialize the whole population on rank 0
        population = (Wolf *)malloc(g_pop_size * sizeof(Wolf));
        if (!population) {
            perror("Error allocating memory for population");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        initialize_population(population, g_pop_size, g_dimension,
                             info->lower_bound, info->upper_bound);
        sort_population(population, g_pop_size);
        alpha = population[0];
        beta  = population[1];
        delta = population[2];
    }

    // Prepare Scatter
    int *sendcounts = (int *)malloc(size * sizeof(int));
    int *displs     = (int *)malloc(size * sizeof(int));
    {
        int offset = 0;
        for (int i = 0; i < size; i++) {
            int portion = g_pop_size / size + ((i < (g_pop_size % size)) ? 1 : 0);
            sendcounts[i] = portion * sizeof(Wolf);
            displs[i]     = offset;
            offset       += sendcounts[i];
        }
    }

    // First Scatter
    MPI_Scatterv(population, sendcounts, displs, MPI_BYTE,
                 local_pop, local_size * sizeof(Wolf),
                 MPI_BYTE, 0, MPI_COMM_WORLD);

    // For storing alpha's fitness, written to file after all iterations
    double *alpha_history = NULL;
    if (rank == 0) {
        alpha_history = (double *)malloc(g_max_iter * sizeof(double));
        if (!alpha_history) {
            fprintf(stderr, "Error allocating memory for alpha_history.\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }

    double start_time = MPI_Wtime();

    // Set sync interval to 2 (only sync every 2 generations)
    int sync_interval = 1;

    for (int iter = 1; iter <= g_max_iter; iter++) {
        // 1) Each process updates local fitness
        for (int i = 0; i < local_size; i++) {
            local_pop[i].fitness = info->function(local_pop[i].position, g_dimension);
        }

        // Only sync (Gather+sort+Scatter) every "sync_interval" generations
        if (iter % sync_interval == 0) {
            // Gather
            MPI_Gatherv(local_pop, local_size * sizeof(Wolf), MPI_BYTE,
                        population, sendcounts, displs, MPI_BYTE,
                        0, MPI_COMM_WORLD);

            if (rank == 0) {
                sort_population(population, g_pop_size);
                alpha = population[0];
                beta  = population[1];
                delta = population[2];

                // Standard GWO position update
                double t = (double)iter / g_max_iter; 
                double a = 2.0 - 2.0 * t;

                for (int i = 0; i < g_pop_size; i++) {
                    for (int d = 0; d < g_dimension; d++) {
                        double r1 = (double)rand() / RAND_MAX;
                        double r2 = (double)rand() / RAND_MAX;

                        double A1 = 2.0 * a * r1 - a;
                        double C1 = 2.0 * r2;
                        double D_alpha = fabs(C1 * alpha.position[d] - population[i].position[d]);
                        double X1 = alpha.position[d] - A1 * D_alpha;

                        r1 = (double)rand() / RAND_MAX;
                        r2 = (double)rand() / RAND_MAX;
                        double A2 = 2.0 * a * r1 - a;
                        double C2 = 2.0 * r2;
                        double D_beta = fabs(C2 * beta.position[d] - population[i].position[d]);
                        double X2 = beta.position[d] - A2 * D_beta;

                        r1 = (double)rand() / RAND_MAX;
                        r2 = (double)rand() / RAND_MAX;
                        double A3 = 2.0 * a * r1 - a;
                        double C3 = 2.0 * r2;
                        double D_delta = fabs(C3 * delta.position[d] - population[i].position[d]);
                        double X3 = delta.position[d] - A3 * D_delta;

                        population[i].position[d] = (X1 + X2 + X3) / 3.0;
                    }
                }

                alpha_history[iter - 1] = alpha.fitness;
            }

            // Broadcast updated alpha,beta,delta
            MPI_Bcast(&alpha, 1, MPI_WOLF, 0, MPI_COMM_WORLD);
            MPI_Bcast(&beta,  1, MPI_WOLF, 0, MPI_COMM_WORLD);
            MPI_Bcast(&delta, 1, MPI_WOLF, 0, MPI_COMM_WORLD);

            // Scatter updated population
            MPI_Scatterv(population, sendcounts, displs, MPI_BYTE,
                         local_pop, local_size * sizeof(Wolf),
                         MPI_BYTE, 0, MPI_COMM_WORLD);
        } else {
            // Non-sync iteration: rank 0 just records alpha if desired
            if (rank == 0) {
                alpha_history[iter - 1] = alpha.fitness;
            }
        }
    }

    double end_time = MPI_Wtime();

    // rank=0 writes logs
    if (rank == 0) {
        double algorithm_time = end_time - start_time;
        write_performance_log(performance_file, algorithm_time);

        // One-time write alpha_history
        FILE *fp = fopen(convergence_file, "w");
        if (fp) {
            for (int i = 0; i < g_max_iter; i++) {
                fprintf(fp, "%d %.6f\n", i + 1, alpha_history[i]);
            }
            fclose(fp);
        } else {
            fprintf(stderr, "Warning: Failed to open file for convergence data.\n");
        }

        free(alpha_history);
        free(population);
    }

    free(local_pop);
    free(sendcounts);
    free(displs);

    MPI_Type_free(&MPI_WOLF);
    MPI_Finalize();
    return 0;
}
