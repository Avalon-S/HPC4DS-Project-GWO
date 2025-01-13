#include "GWO.h"


/* Global variables are defined here and assigned default values ​​*/
int g_dimension = MAX_DIM;
int g_pop_size = MAX_POP_SIZE;
int g_max_iter = MAX_ITER;

/* Initialize the population randomly in the interval [-100, 100] */
void initialize_population(Wolf *population, int pop_size, int dim, double lower_bound, double upper_bound)
{
    srand((unsigned)time(NULL));
    for (int i = 0; i < pop_size; i++) {
        for (int d = 0; d < dim; d++) {
            double randVal = (double)rand() / RAND_MAX; // [0,1]
            population[i].position[d] = lower_bound + randVal * (upper_bound - lower_bound);
        }
        population[i].fitness = 1e30; // Initialize to a larger fitness value
    }
}

/* Simple bubble sort, sorting fitness from small to large (best first) */
void sort_population(Wolf *population, int pop_size)
{
    for (int i = 0; i < pop_size - 1; i++)
    {
        for (int j = 0; j < pop_size - i - 1; j++)
        {
            if (population[j].fitness > population[j + 1].fitness)
            {
                Wolf temp = population[j];
                population[j] = population[j + 1];
                population[j + 1] = temp;
            }
        }
    }
}

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

void create_directory_recursively(const char *dir_path) {
    char temp_path[512];
    snprintf(temp_path, sizeof(temp_path), "%s", dir_path);
    char *path_copy = strdup(temp_path);
    if (!path_copy) {
        perror("Error allocating memory for path_copy");
        exit(EXIT_FAILURE);
    }
    char *parent_dir = dirname(path_copy);

    struct stat st = {0};
    if (stat(parent_dir, &st) == -1) {
        create_directory_recursively(parent_dir);
    }

    if (stat(dir_path, &st) == -1) {
        if (mkdir(dir_path, 0700) != 0 && errno != EEXIST) {
            perror("Error creating directory");
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


 

/* Write convergence data to file (append mode) */
void write_convergence_to_file(const char *filename, int iter, double best_fitness)
{
    FILE *fp = fopen(filename, "a");
    if (fp)
    {
        fprintf(fp, "%d %f\n", iter, best_fitness);
        fclose(fp);
    }
}

/* Write performance log (time and other information) to file (append mode) */
void write_performance_log(const char *filename, double elapsed_time)
{
    FILE *fp = fopen(filename, "a");
    if (fp)
    {
        fprintf(fp, "Execution time: %f seconds\n", elapsed_time);
        fclose(fp);
    }
}

// Function to map function name to the corresponding function pointer
double (*get_test_function(const char *test_function_name))(double *, int) {
    if (strcmp(test_function_name, "sphere") == 0) return sphere;
    if (strcmp(test_function_name, "schwefel_2_22") == 0) return schwefel_2_22;
    if (strcmp(test_function_name, "schwefel_1_2") == 0) return schwefel_1_2;
    if (strcmp(test_function_name, "infinity_norm") == 0) return infinity_norm;
    if (strcmp(test_function_name, "rosenbrock") == 0) return rosenbrock;

    fprintf(stderr, "Unknown test function: %s\n", test_function_name);
    exit(EXIT_FAILURE);
}

// Custom strdup implementation
char *my_strdup(const char *s) {
    if (s == NULL) return NULL; 
    size_t len = strlen(s) + 1; 
    char *copy = malloc(len);   
    if (copy) {
        memcpy(copy, s, len);   
    }
    return copy;
}