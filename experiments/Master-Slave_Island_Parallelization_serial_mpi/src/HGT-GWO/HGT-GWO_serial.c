#include "../common/GWO.h"
#include <mpi.h>

// Historical position arrays
static double *previous_best_pos = NULL;
static double *previous_positions = NULL;

// Weights for HGT-GWO
double alpha_weight = 0.1;
double beta_weight  = 0.1;

// Get the top 3 wolves in the population
static void get_top3(Wolf *population, int pop_size) {
    // Directly sort and take the top 3
    sort_population(population, pop_size);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        fprintf(stderr, "Usage: %s <test_function_name> <dimension>\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    char *test_function_name = argv[1];
    g_dimension = atoi(argv[2]);

    if (g_dimension <= 0) {
        fprintf(stderr, "Error: Invalid dimension.\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    // Retrieve test function information
    TestFunctionInfo *info = get_test_function_info(test_function_name);
    if (!info) {
        fprintf(stderr, "Error: Unknown test function: %s\n", test_function_name);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    // Allocate historical arrays for HGT-GWO
    previous_best_pos  = (double*)malloc(g_dimension * sizeof(double));
    previous_positions = (double*)malloc(g_pop_size * g_dimension * sizeof(double));
    if (!previous_best_pos || !previous_positions) {
        perror("Error allocating memory for previous_* arrays");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    // Population
    Wolf *population = (Wolf*)malloc(g_pop_size * sizeof(Wolf));
    if (!population) {
        perror("Error allocating memory for population");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    srand(12345);
    initialize_population(population, g_pop_size, g_dimension, info->lower_bound, info->upper_bound);
    sort_population(population, g_pop_size);

    Wolf alpha = population[0];
    Wolf beta = population[1];
    Wolf delta = population[2];

    // Record the previous best solution position
    memcpy(previous_best_pos, alpha.position, g_dimension * sizeof(double));
    for (int i = 0; i < g_pop_size; i++) {
        memcpy(&previous_positions[i * g_dimension], population[i].position, g_dimension * sizeof(double));
    }

    double *alpha_history = (double*)malloc(g_max_iter * sizeof(double));
    if (!alpha_history) {
        fprintf(stderr, "Error allocating memory for alpha_history.\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

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
             "%s/convergence_HGT-GWO_serial.txt", convergence_dir);
    snprintf(performance_file, sizeof(performance_file),
             "%s/performance_log_HGT-GWO_serial.txt", performance_dir);
    double start_time = MPI_Wtime();
    double io_time = 0.0;

    // Main loop
    for (int iter = 1; iter <= g_max_iter; iter++) {
        // Calculate fitness
        for (int i = 0; i < g_pop_size; i++) {
            population[i].fitness = info->function(population[i].position, g_dimension);
        }

        // Sort and update alpha, beta, delta
        get_top3(population, g_pop_size);
        alpha = population[0];
        beta = population[1];
        delta = population[2];

        // Record convergence information
        alpha_history[iter - 1] = alpha.fitness;

        // Update positions (HGT-GWO logic)
        double a = 2.0 - 2.0 * ((double)iter / g_max_iter);
        for (int i = 0; i < g_pop_size; i++) {
            for (int d = 0; d < g_dimension; d++) {
                double r1 = (double)rand() / RAND_MAX;
                double r2 = (double)rand() / RAND_MAX;
                double A1 = 2.0 * a * r1 - a;
                double C1 = 2.0 * r2;
                double D_alpha = fabs(C1 * alpha.position[d] - population[i].position[d]);
                double X1 = alpha.position[d] - A1 * D_alpha;

                double hist_guidance = previous_best_pos[d];
                double trend_adjust = population[i].position[d] - previous_positions[i * g_dimension + d];

                population[i].position[d] =
                    (1.0 - alpha_weight - beta_weight) * X1
                    + alpha_weight * hist_guidance
                    + beta_weight * trend_adjust;
            }
        }

        // Update historical positions
        memcpy(previous_best_pos, alpha.position, g_dimension * sizeof(double));
        for (int i = 0; i < g_pop_size; i++) {
            memcpy(&previous_positions[i * g_dimension], population[i].position, g_dimension * sizeof(double));
        }
    }

    double end_time = MPI_Wtime();

    // Write logs
    double total_time = end_time - start_time;
    double algorithm_time = total_time;// - io_time;

    FILE *fp_conv = fopen(convergence_file, "w");
    if (fp_conv) {
        for (int i = 0; i < g_max_iter; i++) {
            fprintf(fp_conv, "%d %.6f\n", i + 1, alpha_history[i]);
        }
        fclose(fp_conv);
    } else {
        fprintf(stderr, "Warning: failed to open %s\n", convergence_file);
    }

    write_performance_log(performance_file, algorithm_time);

    // Cleanup
    free(alpha_history);
    free(previous_best_pos);
    free(previous_positions);
    free(population);

    MPI_Finalize();
    return 0;
}
