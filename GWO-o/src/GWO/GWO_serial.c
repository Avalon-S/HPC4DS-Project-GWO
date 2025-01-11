#include "../common/GWO.h"

int main(int argc, char *argv[])
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <test_function_name> <dimension>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    char *test_function_name = argv[1];
    g_dimension = atoi(argv[2]);

    if (g_dimension <= 0) {
        fprintf(stderr, "Error: Dimension must be a positive integer.\n");
        exit(EXIT_FAILURE);
    }

    srand(12345);

    // g_pop_size = 200;
    // g_max_iter = 500;

    // Retrieve test function info
    TestFunctionInfo *info = get_test_function_info(test_function_name);
    if (!info) {
        fprintf(stderr, "Failed to find the specified test function: %s\n", test_function_name);
        exit(EXIT_FAILURE);
    }

    // Allocate population
    Wolf *population = (Wolf *)malloc(sizeof(Wolf) * g_pop_size);
    if (!population) {
        fprintf(stderr, "Error: Memory allocation failed for population.\n");
        exit(EXIT_FAILURE);
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
    clock_t start = clock();

    // alpha_history to store alpha.fitness each generation
    double *alpha_history = (double *)malloc(g_max_iter * sizeof(double));
    if (!alpha_history) {
        fprintf(stderr, "Error: Memory allocation failed for alpha_history.\n");
        free(population);
        exit(EXIT_FAILURE);
    }

    // We introduce a sync_interval controlling how often (e.g., every 2 generations)
    // we actually do the "sort + position updates."
    int sync_interval = 1;

    // Main GWO loop
    Wolf alpha, beta, delta;
    for (int iter = 1; iter <= g_max_iter; iter++) {
        // Always compute fitness
        for (int i = 0; i < g_pop_size; i++) {
            population[i].fitness = info->function(population[i].position, g_dimension);
        }

        if (iter % sync_interval == 1) {
            // For the first iteration, or if we want to ensure alpha is defined,
            // let's do the usual updates on iteration 1 as well:
            if (iter == 1) {
                // Sort + alpha/beta/delta + position update
                sort_population(population, g_pop_size);
                alpha = population[0];
                beta  = population[1];
                delta = population[2];

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
                alpha_history[iter - 1] = alpha.fitness;
            } else {
                // If not iteration 1 but mod sync_interval=1 => skip update
                // Just carry over the alpha fitness from previous iteration
                alpha_history[iter - 1] = alpha_history[iter - 2];
            }
        }
        else if (iter % sync_interval == 0) {
            // Do the usual "sort + alpha/beta/delta + position update"
            sort_population(population, g_pop_size);
            alpha = population[0];
            beta  = population[1];
            delta = population[2];

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
            alpha_history[iter - 1] = alpha.fitness;
        }
    }

    clock_t end = clock();
    double total_time = (double)(end - start) / CLOCKS_PER_SEC;

    double algorithm_time = total_time;

    // Write out alpha_history at once
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
    write_performance_log(performance_file, algorithm_time);

    free(alpha_history);
    free(population);
    return 0;
}
