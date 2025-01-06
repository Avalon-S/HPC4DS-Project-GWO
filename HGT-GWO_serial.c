#include "../common/GWO.h"

static double previous_best_pos[MAX_DIM] = {10.0};
static double previous_positions[MAX_POP_SIZE][MAX_DIM] = {{10.0}};

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
    int dim = 512;
    int pop_size = 50;
    int max_iter = 1000;
    double lb = -10.0;
    double ub = 10.0;
    g_dimension = dim;
    g_pop_size = pop_size;
    g_max_iter = max_iter;

    /* Check and create convergence log and performance log directories */
    create_directory_if_not_exists("/home/yuhang.jiang/Project/data/convergence/");
    create_directory_if_not_exists("/home/yuhang.jiang/Project/data/performance_logs/");

    Wolf *population = (Wolf *)malloc(sizeof(Wolf) * pop_size);
    initialize_population(population, pop_size, dim, lb, ub);

    /* Record the initial best position and each individual's previous position */
    sort_population(population, pop_size);
    for (int d = 0; d < dim; d++) {
        previous_best_pos[d] = population[0].position[d];
        for (int i = 0; i < pop_size; i++) {
            previous_positions[i][d] = population[i].position[d];
        }
    }

    double alpha_weight = 0.1; // α
    double beta_weight = 0.1;  // β

    clock_t start = clock();
    for (int iter = 1; iter <= max_iter; iter++) {
        // 1. Calculate fitness
        for (int i = 0; i < pop_size; i++) {
            population[i].fitness = evaluate_fitness(population[i].position, dim);
        }

        // 2. Sort to find alpha, beta, delta
        sort_population(population, pop_size);
        Wolf alpha = population[0];
        Wolf beta = population[1];
        Wolf delta = population[2];

        // 3. Standard GWO update (first calculate standard_update_i)
        double a = 2.0 - (2.0 * iter / max_iter);

        // Save the "standard update" result for this iteration
        double standard_update[MAX_POP_SIZE][MAX_DIM];

        for (int i = 0; i < pop_size; i++) {
            for (int d = 0; d < dim; d++) {
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

                standard_update[i][d] = (X1 + X2 + X3) / 3.0;
            }
        }

        // 4. Optimized update formula
        for (int i = 0; i < pop_size; i++) {
            for (int d = 0; d < dim; d++) {
                double standard_part = standard_update[i][d];
                double hist_guidance = previous_best_pos[d];
                double trend_adjust = population[i].position[d] - previous_positions[i][d];

                population[i].position[d] =
                    (1 - alpha_weight - beta_weight) * standard_part
                    + alpha_weight * hist_guidance
                    + beta_weight * trend_adjust;
            }
        }

        // 5. Update previous_best_pos, previous_positions
        //    Before starting the new iteration, record the best solution and individual positions
        for (int d = 0; d < dim; d++) {
            previous_best_pos[d] = alpha.position[d];
        }
        for (int i = 0; i < pop_size; i++) {
            for (int d = 0; d < dim; d++) {
                previous_positions[i][d] = population[i].position[d];
            }
        }

        // 6. Record convergence data
        write_convergence_to_file(
            "/home/yuhang.jiang/Project/data/convergence/HGT-GWO_serial_convergence.txt",
            iter,
            alpha.fitness
        );
    }

    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    write_performance_log("/home/yuhang.jiang/Project/data/performance_logs/HGT-GWO_serial_logs.txt", elapsed);

    free(population);
    return 0;
}
