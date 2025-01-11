#include "../common/GWO.h"

// Global arrays for historical positions
static double *previous_best_pos = NULL;   // Best wolf's position from previous iteration
static double *previous_positions = NULL;  // Each individual's position from previous iteration

int main(int argc, char *argv[])
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <test_function_id> <dimension>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    // Parse arguments
    char *test_function_name = argv[1];
    int dim = atoi(argv[2]);
    if (dim <= 0) {
        fprintf(stderr, "Error: Dimension must be a positive integer.\n");
        exit(EXIT_FAILURE);
    }

    // Initialize parameters
    // int pop_size = 200;
    // int max_iter = 500;
    // g_dimension = dim;
    // g_pop_size  = pop_size;
    // g_max_iter  = max_iter;

    // Dynamically allocate previous_best_pos and previous_positions
    previous_best_pos = (double *)malloc(dim * sizeof(double));
    if (!previous_best_pos) {
        perror("Error allocating memory for previous_best_pos");
        exit(EXIT_FAILURE);
    }

    previous_positions = (double *)malloc(g_pop_size * dim * sizeof(double));
    if (!previous_positions) {
        perror("Error allocating memory for previous_positions");
        free(previous_best_pos);
        exit(EXIT_FAILURE);
    }

    // Get test function information
    TestFunctionInfo *info = get_test_function_info(test_function_name);
    if (!info) {
        fprintf(stderr, "Failed to find the specified test function: %s\n", test_function_name);
        free(previous_best_pos);
        free(previous_positions);
        exit(EXIT_FAILURE);
    }

    // Prepare directories and file paths
    char convergence_dir[256], performance_dir[256];
    generate_directory_path(convergence_dir, sizeof(convergence_dir),
                            "/home/yuhang.jiang/Project/data/convergence",
                            test_function_name, dim, "1_core");
    generate_directory_path(performance_dir, sizeof(performance_dir),
                            "/home/yuhang.jiang/Project/data/performance_logs",
                            test_function_name, dim, "1_core");

    create_directory_if_not_exists(convergence_dir);
    create_directory_if_not_exists(performance_dir);

    char convergence_file[256], performance_file[256];
    snprintf(convergence_file, sizeof(convergence_file),
             "%s/convergence_HGT-GWO_serial.txt", convergence_dir);
    snprintf(performance_file, sizeof(performance_file),
             "%s/performance_log_HGT-GWO_serial.txt", performance_dir);

    // Initialize population
    Wolf *population = (Wolf *)malloc(sizeof(Wolf) * g_pop_size);
    if (!population) {
        perror("Error allocating memory for population");
        free(previous_best_pos);
        free(previous_positions);
        exit(EXIT_FAILURE);
    }

    srand(12345);
    initialize_population(population, g_pop_size, dim,
                          info->lower_bound, info->upper_bound);

    // Sort and record initial best + positions
    sort_population(population, g_pop_size);
    for (int d = 0; d < dim; d++) {
        previous_best_pos[d] = population[0].position[d];
    }
    for (int i = 0; i < g_pop_size; i++) {
        for (int d = 0; d < dim; d++) {
            previous_positions[i * dim + d] = population[i].position[d];
        }
    }

    // HGT-GWO parameters
    double alpha_weight = 0.1; // α
    double beta_weight  = 0.1; // β

    // To reduce I/O overhead, alpha fitness is temporarily stored in memory and written out at once.
    double *alpha_history = (double *)malloc(g_max_iter * sizeof(double));
    if (!alpha_history) {
        fprintf(stderr, "Error: Failed to allocate memory for alpha_history.\n");
        free(population);
        free(previous_best_pos);
        free(previous_positions);
        exit(EXIT_FAILURE);
    }

    // Start timing
    clock_t start = clock();
    double io_time = 0.0; // Time spent on file I/O

    // Main iteration
    for (int iter = 1; iter <= g_max_iter; iter++) {
        // 1. Calculate fitness
        for (int i = 0; i < g_pop_size; i++) {
            population[i].fitness = info->function(population[i].position, dim);
        }

        // 2. Sort, find alpha, beta, delta
        sort_population(population, g_pop_size);
        Wolf alpha = population[0];
        Wolf beta  = population[1];
        Wolf delta = population[2];

        // 3. Standard GWO update
        double a = 2.0 - (2.0 * iter / g_max_iter);
        static double standard_update[MAX_POP_SIZE][MAX_DIM]; 

        for (int i = 0; i < g_pop_size; i++) {
            for (int d = 0; d < dim; d++) {
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

                standard_update[i][d] = (X1 + X2 + X3) / 3.0;
            }
        }

        // 4. HGT-GWO update formula
        for (int i = 0; i < g_pop_size; i++) {
            for (int d = 0; d < dim; d++) {
                double standard_part = standard_update[i][d];
                double hist_guidance = previous_best_pos[d];
                double trend_adjust  = population[i].position[d] - previous_positions[i * dim + d];

                population[i].position[d] =
                    (1.0 - alpha_weight - beta_weight) * standard_part
                    + alpha_weight * hist_guidance
                    + beta_weight  * trend_adjust;
            }
        }

        // 5. Update historical info
        for (int d = 0; d < dim; d++) {
            previous_best_pos[d] = alpha.position[d];
        }
        for (int i = 0; i < g_pop_size; i++) {
            for (int d = 0; d < dim; d++) {
                previous_positions[i * dim + d] = population[i].position[d];
            }
        }

        // Store alpha's fitness into memory
        alpha_history[iter - 1] = alpha.fitness;
    }

    // End timing
    clock_t end = clock();
    double total_time = (double)(end - start) / CLOCKS_PER_SEC;
    double algorithm_time = total_time - io_time; 

    // Write out convergence data at one time
    {
        clock_t io_start = clock();
        FILE *fp_convergence = fopen(convergence_file, "w");
        if (fp_convergence) {
            for (int iter = 1; iter <= g_max_iter; iter++) {
                fprintf(fp_convergence, "%d %.6f\n", iter, alpha_history[iter - 1]);
            }
            fclose(fp_convergence);
        } else {
            fprintf(stderr, "Warning: failed to open file for convergence: %s\n", convergence_file);
        }
        clock_t io_end = clock();
        io_time += (double)(io_end - io_start) / CLOCKS_PER_SEC;
    }


    double final_algorithm_time = total_time - io_time;

    // Write performance log
    write_performance_log(performance_file, final_algorithm_time);

    // Free memory
    free(alpha_history);
    free(population);
    free(previous_best_pos);
    free(previous_positions);

    return 0;
}
