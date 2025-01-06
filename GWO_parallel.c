#include "../common/GWO.h"
#include <mpi.h>
#include <errno.h>
#include <libgen.h>
#include <sys/stat.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

// Macro definition for base data path
#define DATA_PATH "/home/yuhang.jiang/Project/data"

// Common function: Check and create directory if not exists
void create_directory_if_not_exists(const char *dir_path) {
    struct stat st = {0};
    if (stat(dir_path, &st) == -1) {
        if (mkdir(dir_path, 0700) != 0) {
            perror("Error creating directory");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    // Ensure enough arguments are provided
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <test_function_id> <dimension> <num_cores>\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    char *test_function_name = argv[1];
    int dim = atoi(argv[2]);
    int num_cores = atoi(argv[3]);

    if (dim <= 0 || num_cores <= 0) {
        fprintf(stderr, "Error: Dimension and core count must be positive integers.\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    g_dimension = dim;
    g_pop_size = 50;
    g_max_iter = 1000;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (num_cores != size) {
        fprintf(stderr, "Error: The number of cores specified does not match the MPI size.\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    // Dynamically allocate local population
    int local_size = g_pop_size / size + (rank < g_pop_size % size ? 1 : 0);
    Wolf *local_pop = (Wolf *)malloc(sizeof(Wolf) * local_size);
    if (!local_pop) {
        perror("Error allocating memory for local population");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    // Root process initializes population
    Wolf *population = NULL;
    int *sendcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));
    if (rank == 0) {
        population = (Wolf *)malloc(sizeof(Wolf) * g_pop_size);
        if (!population) {
            perror("Error allocating memory for population on root process");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        initialize_population(population, g_pop_size, g_dimension, -100.0, 100.0);

        // Calculate scatter parameters
        int offset = 0;
        for (int i = 0; i < size; i++) {
            sendcounts[i] = (g_pop_size / size) + (i < g_pop_size % size ? 1 : 0);
            sendcounts[i] *= sizeof(Wolf);
            displs[i] = offset;
            offset += sendcounts[i];
        }
    }

    // Scatter population
    MPI_Scatterv(population, sendcounts, displs, MPI_BYTE,
                 local_pop, local_size * sizeof(Wolf), MPI_BYTE,
                 0, MPI_COMM_WORLD);

    // Prepare directories and file paths
    char convergence_dir[256];
    char performance_dir[256];
    char convergence_file[256];
    char performance_file[256];
    if (rank == 0) {
        snprintf(convergence_dir, sizeof(convergence_dir), "%s/convergence/%s/%dD/%d_cores", DATA_PATH, test_function_name, g_dimension, num_cores);
        snprintf(performance_dir, sizeof(performance_dir), "%s/performance_logs/%s/%dD/%d_cores", DATA_PATH, test_function_name, g_dimension, num_cores);
        create_directory_if_not_exists(convergence_dir);
        create_directory_if_not_exists(performance_dir);

        snprintf(convergence_file, sizeof(convergence_file), "%s/convergence_mpi.txt", convergence_dir);
        snprintf(performance_file, sizeof(performance_file), "%s/performance_log_mpi.txt", performance_dir);
    }

    double start = MPI_Wtime();
    for (int iter = 1; iter <= g_max_iter; iter++) {
        // Recalculate fitness for each individual locally
        for (int i = 0; i < local_size; i++) {
            local_pop[i].fitness = evaluate_fitness(local_pop[i].position, g_dimension);
        }

        // Gather all individuals back to the root process
        MPI_Gatherv(local_pop, local_size * sizeof(Wolf), MPI_BYTE,
                    population, sendcounts, displs, MPI_BYTE,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            // Sort and update positions
            sort_population(population, g_pop_size);
            Wolf alpha = population[0];
            Wolf beta = population[1];
            Wolf delta = population[2];

            double a = 2.0 - (2.0 * iter / g_max_iter);
            for (int i = 0; i < g_pop_size; i++) {
                for (int d = 0; d < g_dimension; d++) {
                    double r1 = (double)rand() / RAND_MAX;
                    double r2 = (double)rand() / RAND_MAX;
                    double A1 = 2 * a * r1 - a;
                    double C1 = 2 * r2;
                    double D_alpha = fabs(C1 * alpha.position[d] - population[i].position[d]);
                    double X1 = alpha.position[d] - A1 * D_alpha;

                    r1 = (double)rand() / RAND_MAX;
                    r2 = (double)rand() / RAND_MAX;
                    double A2 = 2 * a * r1 - a;
                    double C2 = 2 * r2;
                    double D_beta = fabs(C2 * beta.position[d] - population[i].position[d]);
                    double X2 = beta.position[d] - A2 * D_beta;

                    r1 = (double)rand() / RAND_MAX;
                    r2 = (double)rand() / RAND_MAX;
                    double A3 = 2 * a * r1 - a;
                    double C3 = 2 * r2;
                    double D_delta = fabs(C3 * delta.position[d] - population[i].position[d]);
                    double X3 = delta.position[d] - A3 * D_delta;

                    population[i].position[d] = (X1 + X2 + X3) / 3.0;
                }
            }

            // Write convergence data
            write_convergence_to_file(convergence_file, iter, population[0].fitness);
        }

        // Broadcast updated population
        MPI_Bcast(population, g_pop_size * sizeof(Wolf), MPI_BYTE, 0, MPI_COMM_WORLD);

        // Redistribute population
        MPI_Scatterv(population, sendcounts, displs, MPI_BYTE,
                     local_pop, local_size * sizeof(Wolf), MPI_BYTE,
                     0, MPI_COMM_WORLD);
    }
    double end = MPI_Wtime();

    // Write performance logs
    if (rank == 0) {
        double elapsed = end - start;
        write_performance_log(performance_file, elapsed);
        free(population);
    }

    free(local_pop);
    free(sendcounts);
    free(displs);
    MPI_Finalize();
    return 0;
}
