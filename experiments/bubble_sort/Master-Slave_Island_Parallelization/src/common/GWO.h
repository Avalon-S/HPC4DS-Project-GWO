#ifndef GWO_H
#define GWO_H
#define _USE_MATH_DEFINES
#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h> // for rand_r, malloc, etc.
#include <string.h> // For using memcpy()
#include <sys/stat.h>
#include <sys/types.h>
#include <stddef.h>   // for offsetof, HGT-GWO_hybrid.c
#include <errno.h>
#include <libgen.h> // For dirname


/* ------------------ General macro definition ------------------ */
#define MAX_POP_SIZE 1000 
#define MAX_DIM 1024
#define MAX_ITER 500

/* ------------------ Data Structure ------------------ */
typedef struct {
    double position[MAX_DIM];  // Current individual position
    double fitness;            // The fitness of the current individual
} Wolf;

typedef struct {
    const char *name;          
    double lower_bound;        
    double upper_bound;        
    double solution;           
    double (*function)(double *, int);  
} TestFunctionInfo;

/* ------------------ Global variable declaration ------------------ */
extern int g_dimension;     
extern int g_pop_size;        
extern int g_max_iter;       

/* ------------------ Function declaration ------------------ */

/* Get the function pointer from the function name */
double (*get_test_function(const char *test_function_name))(double *, int);

/* Get complete information of the test function (name, boundary, global optimal solution, etc.) */
TestFunctionInfo *get_test_function_info(const char *test_function_name);

/* string copy */
char *my_strdup(const char *s);


/* From common_functions.c */
void initialize_population(Wolf *population, int pop_size, int dim, double lower_bound, double upper_bound);
void sort_population(Wolf *population, int pop_size);
void create_directory_if_not_exists(const char *dir_path);
void create_directory_recursively(const char *dir_path);
void generate_directory_path(char *buffer, size_t size, const char *base_path, const char *test_function_name, int dimension, const char *core_count);
void write_convergence_to_file(const char *filename, int iter, double best_fitness);
void write_performance_log(const char *filename, double elapsed_time);


/* From test_functions.c */
double sphere(double *x, int dim);
double schwefel_2_22(double *x, int dim);
double schwefel_1_2(double *x, int dim);
double infinity_norm(double *x, int dim);
double rosenbrock(double *x, int dim);

extern TestFunctionInfo test_functions[];
extern const int num_test_functions;

#endif
