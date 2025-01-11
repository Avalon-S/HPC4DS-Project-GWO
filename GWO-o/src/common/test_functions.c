#define _USE_MATH_DEFINES
#include "GWO.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846 // Manually define the value of π
#endif

#ifndef M_E
#define M_E 2.71828182845904523536 // Manually define the value of e
#endif

TestFunctionInfo test_functions[] = {
    {"F1", -100.0, 100.0, 0.0, sphere},
    {"F2", -100.0, 100.0, 0.0, schwefel_2_22},
    {"F3", -100.0, 100.0, 0.0, schwefel_1_2},
    {"F4", -100.0, 100.0, 0.0, infinity_norm},
    {"F5", -100.0, 100.0, 0.0, rosenbrock},
};

const int num_test_functions = sizeof(test_functions) / sizeof(test_functions[0]);

TestFunctionInfo *get_test_function_info(const char *test_function_name) {
    for (int i = 0; i < num_test_functions; i++) {
        if (strcmp(test_functions[i].name, test_function_name) == 0) {
            return &test_functions[i];
        }
    }
    fprintf(stderr, "Error: Unknown test function: %s\n", test_function_name);
    return NULL;
}

// F1: Sphere Function
double sphere(double *x, int dim) {
    double sum = 0.0;
    for (int i = 0; i < dim; i++) {
        sum += x[i] * x[i];
    }
    return sum;
}

// F2: Schwefel 2.22 Function
double schwefel_2_22(double *x, int dim) {
    double sum_abs = 0.0, prod_abs = 1.0;
    for (int i = 0; i < dim; i++) {
        sum_abs += fabs(x[i]);
        prod_abs *= fabs(x[i]);
    }
    return sum_abs + prod_abs;
}

// F3: Schwefel 1.2 Function
double schwefel_1_2(double *x, int dim) {
    double sum = 0.0;
    for (int i = 0; i < dim; i++) {
        double partial_sum = 0.0;
        for (int j = 0; j <= i; j++) {
            partial_sum += x[j];
        }
        sum += partial_sum * partial_sum;
    }
    return sum;
}

// F4: Infinity Norm Function
double infinity_norm(double *x, int dim) {
    double max_val = fabs(x[0]);
    for (int i = 1; i < dim; i++) {
        if (fabs(x[i]) > max_val) {
            max_val = fabs(x[i]);
        }
    }
    return max_val;
}

// F5: Rosenbrock Function
double rosenbrock(double *x, int dim) {
    double sum = 0.0;
    for (int i = 0; i < dim - 1; i++) {
        sum += 100.0 * pow((x[i + 1] - x[i] * x[i]), 2) + pow((1 - x[i]), 2);
    }
    return sum;
}
