#include "../common/GWO.h"
#include <errno.h>
#include <libgen.h>
#include <sys/stat.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

// Create directories recursively if they don't exist
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

// Helper function to generate directory paths
void generate_directory_path(char *buffer, size_t size, const char *base_path, const char *test_function_name, int dimension, const char *core_count) {
    snprintf(buffer, size, "%s/%s/%dD/%s", base_path, test_function_name, dimension, core_count);
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <test_function_id> <dimension>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    char *test_function_name = argv[1];
    g_dimension = atoi(argv[2]);
    if (g_dimension <= 0) {
        fprintf(stderr, "Error: Dimension must be a positive integer.\n");
        exit(EXIT_FAILURE);
    }

    g_pop_size = 50;
    g_max_iter = 1000;

    TestFunctionInfo *info = get_test_function_info(test_function_name);
    if (!info) {
        fprintf(stderr, "Failed to find the specified test function: %s\n", test_function_name);
        exit(EXIT_FAILURE);
    }

    Wolf *population = (Wolf *)malloc(sizeof(Wolf) * g_pop_size);
    if (!population) {
        fprintf(stderr, "Error: Memory allocation failed for population.\n");
        exit(EXIT_FAILURE);
    }
    initialize_population(population, g_pop_size, g_dimension, info->lower_bound, info->upper_bound);

    char convergence_dir[256];
    char performance_dir[256];
    generate_directory_path(convergence_dir, sizeof(convergence_dir), "/home/yuhang.jiang/Project/data/convergence", test_function_name, g_dimension, "1_core");
    generate_directory_path(performance_dir, sizeof(performance_dir), "/home/yuhang.jiang/Project/data/performance_logs", test_function_name, g_dimension, "1_core");

    create_directory_if_not_exists(convergence_dir);
    create_directory_if_not_exists(performance_dir);

    char convergence_file[256];
    char performance_file[256];
    snprintf(convergence_file, sizeof(convergence_file), "%s/convergence.txt", convergence_dir);
    snprintf(performance_file, sizeof(performance_file), "%s/performance_log.txt", performance_dir);

    clock_t start = clock();
    for (int iter = 1; iter <= g_max_iter; iter++) {
        for (int i = 0; i < g_pop_size; i++) {
            population[i].fitness = info->function(population[i].position, g_dimension);
        }
        sort_population(population, g_pop_size);
        Wolf alpha = population[0];
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
                double D_beta = fabs(C2 * population[1].position[d] - population[i].position[d]);
                double X2 = population[1].position[d] - A2 * D_beta;

                r1 = (double)rand() / RAND_MAX;
                r2 = (double)rand() / RAND_MAX;
                double A3 = 2 * a * r1 - a;
                double C3 = 2 * r2;
                double D_delta = fabs(C3 * population[2].position[d] - population[i].position[d]);
                double X3 = population[2].position[d] - A3 * D_delta;

                population[i].position[d] = (X1 + X2 + X3) / 3.0;
            }
        }
        write_convergence_to_file(convergence_file, iter, alpha.fitness);
    }

    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    write_performance_log(performance_file, elapsed);

    free(population);
    return 0;
}
