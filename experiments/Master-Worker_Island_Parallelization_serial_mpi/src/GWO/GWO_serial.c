#include "../common/GWO.h"
#include <mpi.h>

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv); // Initialize MPI for timing purposes

    if (argc < 3) {
        fprintf(stderr, "Usage: %s <test_function_name> <dimension>\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    char *test_function_name = argv[1];
    g_dimension = atoi(argv[2]);

    if (g_dimension <= 0) {
        fprintf(stderr, "Error: Dimension must be a positive integer.\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    srand(12345);

    // Retrieve test function info
    TestFunctionInfo *info = get_test_function_info(test_function_name);
    if (!info) {
        fprintf(stderr, "Failed to find the specified test function: %s\n", test_function_name);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    // Allocate population
    Wolf *population = (Wolf *)malloc(sizeof(Wolf) * g_pop_size);
    if (!population) {
        fprintf(stderr, "Error: Memory allocation failed for population.\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    // Initialize population
    initialize_population(population, g_pop_size, g_dimension,
                          info->lower_bound, info->upper_bound);

    // Prepare output directories and files
    char convergence_dir[256];
    char performance_dir[256];
    generate_directory_path(convergence_dir, sizeof(convergence_dir),
                            "/home/yuhang.jiang/Project/data/convergence",
                            test_function_name, g_dimension, "1_core");

    generate_directory_path(performance_dir, sizeof(performance_dir),
                            "/home/yuhang.jiang/Project/data/performance_logs",
                            test_function_name, g_dimension, "1_core");

    create_directory_if_not_exists(convergence_dir);
    create_directory_if_not_exists(performance_dir);

    char convergence_file[256];
    char performance_file[256];
    snprintf(convergence_file, sizeof(convergence_file),
             "%s/convergence_GWO_serial.txt", convergence_dir);
    snprintf(performance_file, sizeof(performance_file),
             "%s/performance_log_GWO_serial.txt", performance_dir);

    // Timing
    double start_time = MPI_Wtime();

    // alpha_history to store alpha.fitness each generation
    double *alpha_history = (double *)malloc(g_max_iter * sizeof(double));
    if (!alpha_history) {
        fprintf(stderr, "Error: Memory allocation failed for alpha_history.\n");
        free(population);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    // Main GWO loop
    Wolf alpha, beta, delta;
    for (int iter = 1; iter <= g_max_iter; iter++) {
        // Compute fitness
        for (int i = 0; i < g_pop_size; i++) {
            population[i].fitness = info->function(population[i].position, g_dimension);
        }

        // Sort population and update alpha, beta, delta
        sort_population(population, g_pop_size);
        alpha = population[0];
        beta = population[1];
        delta = population[2];

        // Update positions
        double a = 2.0 - (2.0 * iter / g_max_iter);
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

        // Store alpha fitness for convergence
        alpha_history[iter - 1] = alpha.fitness;
    }

    double end_time = MPI_Wtime();
    double total_time = end_time - start_time;

    // Write alpha history to file
    FILE *fp_convergence = fopen(convergence_file, "w");
    if (fp_convergence) {
        for (int iter = 1; iter <= g_max_iter; iter++) {
            fprintf(fp_convergence, "%d %.6f\n", iter, alpha_history[iter - 1]);
        }
        fclose(fp_convergence);
    } else {
        fprintf(stderr, "Warning: Failed to open %s for writing convergence.\n", convergence_file);
    }

    // Write performance log
    write_performance_log(performance_file, total_time);

    // Cleanup
    free(alpha_history);
    free(population);

    MPI_Finalize(); // Finalize MPI
    return 0;
}
