#define _DEFAULT_SOURCE
#include "../common/GWO.h"
#include <mpi.h>
#include <omp.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// Macro definitions for base paths
#define DATA_PATH "/home/yuhang.jiang/Project/data"

// Create directory if not exists
void create_directory_if_not_exists(const char *dir_path) {
    struct stat st = {0};
    if (stat(dir_path, &st) == -1) {
        if (mkdir(dir_path, 0700) != 0) {
            perror("Error creating directory");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}

// Thread-safe random number generator
double thread_safe_random(unsigned int *seed) {
    return (double)rand_r(seed) / RAND_MAX;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    // Ensure correct number of arguments
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <test_function_name> <dimension> <num_cores>\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    // Parse input arguments
    char *test_function_name = argv[1];
    int dim = atoi(argv[2]);
    int num_cores = atoi(argv[3]);
    g_dimension = dim;
    g_pop_size = 50;   // Default population size
    g_max_iter = 1000; // Default maximum iterations

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (num_cores != size) {
        fprintf(stderr, "Error: Specified number of cores (%d) does not match MPI size (%d).\n", num_cores, size);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    // Get test function information
    TestFunctionInfo *info = get_test_function_info(test_function_name);
    if (info == NULL) {
        fprintf(stderr, "Failed to find the specified test function: %s\n", test_function_name);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    double lb = info->lower_bound;
    double ub = info->upper_bound;

    // Adjust population size for divisibility
    int remainder = g_pop_size % size;
    if (remainder != 0) {
        if (rank == 0) {
            printf("Adjusting population size from %d to %d for divisibility.\n", g_pop_size, g_pop_size + (size - remainder));
        }
        g_pop_size += (size - remainder);
    }
    int local_size = g_pop_size / size;

    // Allocate memory
    Wolf *local_pop = (Wolf *)malloc(sizeof(Wolf) * local_size);
    Wolf *population = (rank == 0) ? (Wolf *)malloc(sizeof(Wolf) * g_pop_size) : NULL;
    Wolf *new_population = (rank == 0) ? (Wolf *)malloc(sizeof(Wolf) * g_pop_size) : NULL;
    if (!local_pop || (rank == 0 && (!population || !new_population))) {
        perror("Error allocating memory");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    // Initialize population (only on root)
    if (rank == 0) {
        initialize_population(population, g_pop_size, g_dimension, lb, ub);
    }

    // Scatter initial population
    MPI_Scatter(population, local_size * sizeof(Wolf), MPI_BYTE,
                local_pop, local_size * sizeof(Wolf), MPI_BYTE,
                0, MPI_COMM_WORLD);

    // Prepare file paths for results
    char convergence_dir[256];
    char performance_dir[256];
    char convergence_file[256];
    char performance_file[256];
    if (rank == 0) {
        snprintf(convergence_dir, sizeof(convergence_dir), "%s/convergence/%s/%dD/%d_cores", DATA_PATH, test_function_name, dim, num_cores);
        snprintf(performance_dir, sizeof(performance_dir), "%s/performance_logs/%s/%dD/%d_cores", DATA_PATH, test_function_name, dim, num_cores);
        create_directory_if_not_exists(convergence_dir);
        create_directory_if_not_exists(performance_dir);

        snprintf(convergence_file, sizeof(convergence_file), "%s/convergence_mpi_openmp.txt", convergence_dir);
        snprintf(performance_file, sizeof(performance_file), "%s/performance_log_mpi_openmp.txt", performance_dir);
    }

    // Prepare random seeds for OpenMP
    int max_threads = omp_get_max_threads();
    unsigned int *base_seeds = (unsigned int *)malloc(sizeof(unsigned int) * max_threads);
    srand((unsigned int)(time(NULL) + rank * 12345));
    for (int t = 0; t < max_threads; t++) {
        base_seeds[t] = rand();
    }

    double start = MPI_Wtime();

    // Main optimization loop
    for (int iter = 1; iter <= g_max_iter; iter++) {
        // Evaluate local fitness
        #pragma omp parallel for
        for (int i = 0; i < local_size; i++) {
            local_pop[i].fitness = info->function(local_pop[i].position, g_dimension);
        }

        // Gather all local populations
        MPI_Gather(local_pop, local_size * sizeof(Wolf), MPI_BYTE,
                   population, local_size * sizeof(Wolf), MPI_BYTE,
                   0, MPI_COMM_WORLD);

        if (rank == 0) {
            // Sort and update positions
            sort_population(population, g_pop_size);
            Wolf alpha = population[0];
            Wolf beta = population[1];
            Wolf delta = population[2];

            double a = 2.0 - (2.0 * iter / g_max_iter);

            #pragma omp parallel for
            for (int i = 0; i < g_pop_size; i++) {
                unsigned int seed = base_seeds[omp_get_thread_num()] + i;
                for (int d = 0; d < g_dimension; d++) {
                    double r1 = thread_safe_random(&seed);
                    double r2 = thread_safe_random(&seed);
                    double A1 = 2.0 * a * r1 - a;
                    double C1 = 2.0 * r2;
                    double D_alpha = fabs(C1 * alpha.position[d] - population[i].position[d]);
                    double X1 = alpha.position[d] - A1 * D_alpha;

                    r1 = thread_safe_random(&seed);
                    r2 = thread_safe_random(&seed);
                    double A2 = 2.0 * a * r1 - a;
                    double C2 = 2.0 * r2;
                    double D_beta = fabs(C2 * beta.position[d] - population[i].position[d]);
                    double X2 = beta.position[d] - A2 * D_beta;

                    r1 = thread_safe_random(&seed);
                    r2 = thread_safe_random(&seed);
                    double A3 = 2.0 * a * r1 - a;
                    double C3 = 2.0 * r2;
                    double D_delta = fabs(C3 * delta.position[d] - population[i].position[d]);
                    double X3 = delta.position[d] - A3 * D_delta;

                    double new_val = (X1 + X2 + X3) / 3.0;
                    if (new_val < lb) new_val = lb;
                    if (new_val > ub) new_val = ub;
                    new_population[i].position[d] = new_val;
                }
            }

            memcpy(population, new_population, sizeof(Wolf) * g_pop_size);

            // Log convergence
            write_convergence_to_file(convergence_file, iter, population[0].fitness);
        }

        // Broadcast updated population
        MPI_Bcast(population, g_pop_size * sizeof(Wolf), MPI_BYTE, 0, MPI_COMM_WORLD);

        // Scatter updated population to local processes
        MPI_Scatter(population, local_size * sizeof(Wolf), MPI_BYTE,
                    local_pop, local_size * sizeof(Wolf), MPI_BYTE,
                    0, MPI_COMM_WORLD);
    }

    double end = MPI_Wtime();

    // Record performance log
    if (rank == 0) {
        double elapsed = end - start;
        write_performance_log(performance_file, elapsed);
    }

    // Free resources
    free(base_seeds);
    free(local_pop);
    if (rank == 0) {
        free(population);
        free(new_population);
    }

    MPI_Finalize();
    return 0;
}
