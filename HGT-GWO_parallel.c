#include "../common/GWO.h"
#include <mpi.h>


// Used to record the position of the "global best individual" from the previous generation (historical guidance). Initialized to 10.0 for demonstration purposes.
static double previous_best_pos[MAX_DIM] = {10.0};

// Common function: Check and create directory if not exists
void create_directory_if_not_exists(const char *dir_path) {
    char temp_path[512];
    snprintf(temp_path, sizeof(temp_path), "%s", dir_path);
    char *path_copy = my_strdup(temp_path);
    if (!path_copy) {
        fprintf(stderr, "Error: Failed to allocate memory for path_copy\n");
        exit(EXIT_FAILURE);
    }
    char *parent_dir = dirname(path_copy);

    struct stat st = {0};
    if (stat(parent_dir, &st) == -1) {
        create_directory_if_not_exists(parent_dir);
    }

    if (stat(dir_path, &st) == -1) {
        if (mkdir(dir_path, 0700) != 0 && errno != EEXIST) {
            perror("Error creating directory");
            fprintf(stderr, "Failed to create directory: %s\n", dir_path);
            free(path_copy);
            exit(EXIT_FAILURE);
        }
    }

    free(path_copy);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int dim = 512;
    int pop_size = 50;
    int max_iter = 1000;
    double lb = -10.0;
    double ub = 10.0;

    // Global variables declared externally in ../common/GWO.h
    g_dimension = dim;
    g_pop_size  = pop_size;
    g_max_iter  = max_iter;

    // Weights corresponding to α and β
    double alpha_weight = 0.1;
    double beta_weight  = 0.1;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Calculate the number of individuals assigned to each process (may not be even)
    int base = pop_size / size;
    int remainder = pop_size % size;
    int local_size = base + ((rank < remainder) ? 1 : 0);

    // Allocate local population
    Wolf *local_pop = (Wolf *)malloc(sizeof(Wolf) * local_size);
    if (local_pop == NULL) {
        perror("Error allocating memory for local population");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    // Allocate global population
    Wolf *population = NULL;
    if (rank == 0) {
        population = (Wolf *)malloc(sizeof(Wolf) * pop_size);
        if (population == NULL) {
            perror("Error allocating memory for population");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        // Initialize population
        initialize_population(population, pop_size, dim, lb, ub);
        // Record the "initial global best position" to previous_best_pos
        for (int d = 0; d < dim; d++) {
            previous_best_pos[d] = population[0].position[d];
        }
    } else {
        // Non-root processes also allocate the same size buffer for Gatherv/Scatterv
        population = (Wolf *)malloc(sizeof(Wolf) * pop_size);
        if (population  == NULL) {
            perror("Error allocating memory for population on non-root process");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }

    // Construct Scatterv/Gatherv parameters
    int *sendcounts = NULL;
    int *displs     = NULL;
    int *recvcounts = NULL;
    int *displs_g   = NULL;

    if (rank == 0) {
        sendcounts = (int*)malloc(size * sizeof(int));
        displs     = (int*)malloc(size * sizeof(int));
        recvcounts = (int*)malloc(size * sizeof(int));
        displs_g   = (int*)malloc(size * sizeof(int));

        int offset = 0;
        for (int i = 0; i < size; i++) {
            int chunk = base + ((i < remainder) ? 1 : 0);
            sendcounts[i] = chunk * sizeof(Wolf);
            displs[i]     = offset;
            offset       += sendcounts[i];
        }
        // Gatherv receive sizes are identical to Scatterv
        offset = 0;
        for (int i = 0; i < size; i++) {
            recvcounts[i] = sendcounts[i];
            displs_g[i]   = displs[i];
        }
    }

    // Initial Scatterv: Distribute initial population
    MPI_Scatterv(
        population,               // Sending buffer (valid only on root)
        sendcounts,               // Byte size sent to each process
        displs,
        MPI_BYTE,
        local_pop,                // Local receiving buffer
        local_size * sizeof(Wolf),
        MPI_BYTE,
        0,
        MPI_COMM_WORLD
    );

    // Save "trend_adjust = X_i(t) - X_i(t-1)"
    Wolf *old_population = NULL;
    if (rank == 0) {
        old_population = (Wolf *)malloc(sizeof(Wolf) * pop_size);
        memcpy(old_population, population, sizeof(Wolf) * pop_size); // Copy initial population
    }

    double start = MPI_Wtime();

    // Begin iterations
    for (int iter = 1; iter <= max_iter; iter++) {
        // Calculate fitness for the local population
        for (int i = 0; i < local_size; i++) {
            local_pop[i].fitness = evaluate_fitness(local_pop[i].position, dim);
        }

        // Gather the local populations back to root
        MPI_Gatherv(
            local_pop,                       // Local data to send
            local_size * sizeof(Wolf),
            MPI_BYTE,
            population,                      // Receiving buffer on root
            recvcounts,
            displs_g,
            MPI_BYTE,
            0,
            MPI_COMM_WORLD
        );

        if (rank == 0) {
            // Sort population and identify Alpha, Beta, Delta
            sort_population(population, pop_size);
            Wolf alpha = population[0];
            Wolf beta  = population[1];
            Wolf delta = population[2];

            double a = 2.0 - (2.0 * iter / max_iter);

            // Update rules using old_population for "trend_adjust"
            for (int i = 0; i < pop_size; i++) {
                for (int d = 0; d < dim; d++) {
                    // Standard GWO update
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

                    double standard_update = (X1 + X2 + X3) / 3.0;

                    // Historical guidance + trend adjustment
                    double hist_guidance = previous_best_pos[d];
                    double trend_adjust  = population[i].position[d] 
                                         - old_population[i].position[d];

                    population[i].position[d] =
                        (1 - alpha_weight - beta_weight) * standard_update
                        + alpha_weight * hist_guidance
                        + beta_weight  * trend_adjust;
                }
            }

            // Update the global best position
            for (int d = 0; d < dim; d++) {
                previous_best_pos[d] = alpha.position[d];
            }

            // Write convergence logs
            create_directory_if_not_exists("/home/yuhang.jiang/Project/data/convergence/");
            write_convergence_to_file(
                "/home/yuhang.jiang/Project/data/convergence/HGT-GWO_parallel_convergence.txt",
                iter,
                alpha.fitness
            );

            // Update old_population with the current generation
            memcpy(old_population, population, sizeof(Wolf) * pop_size);
        }

        // Broadcast updated population from root to all processes
        MPI_Bcast(
            population,
            pop_size * sizeof(Wolf),
            MPI_BYTE,
            0,
            MPI_COMM_WORLD
        );

        // Redistribute updated population to all processes using Scatterv
        MPI_Scatterv(
            population,
            sendcounts,
            displs,
            MPI_BYTE,
            local_pop,
            local_size * sizeof(Wolf),
            MPI_BYTE,
            0,
            MPI_COMM_WORLD
        );
    }

    double end = MPI_Wtime();
    if (rank == 0) {
        create_directory_if_not_exists("/home/yuhang.jiang/Project/data/performance_logs/");
        double elapsed = end - start;
        write_performance_log("/home/yuhang.jiang/Project/data/performance_logs/HGT-GWO_parallel_logs.txt", elapsed);

        // Free dynamically allocated arrays on root
        free(sendcounts);
        free(displs);
        free(recvcounts);
        free(displs_g);
        free(old_population);
        free(population);
    }
    // Free local population
    free(local_pop);

    MPI_Finalize();
    return 0;
}
