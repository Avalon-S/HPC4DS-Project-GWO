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

    // 获取测试函数信息
    TestFunctionInfo *info = get_test_function_info(test_function_name);
    if (!info) {
        fprintf(stderr, "Failed to find the specified test function: %s\n", test_function_name);
        exit(EXIT_FAILURE);
    }

    // 分配种群
    Wolf *population = (Wolf *)malloc(sizeof(Wolf) * g_pop_size);
    if (!population) {
        fprintf(stderr, "Error: Memory allocation failed for population.\n");
        exit(EXIT_FAILURE);
    }

    // 初始化种群
    initialize_population(population, g_pop_size, g_dimension, info->lower_bound, info->upper_bound);

    // 准备输出目录和文件
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
    snprintf(convergence_file, sizeof(convergence_file), "%s/convergence_GWO_serial.txt", convergence_dir);
    snprintf(performance_file, sizeof(performance_file), "%s/performance_log_GWO_serial.txt", performance_dir);

    // 计时
    clock_t start = clock();

    // 这里演示将每一代的 alpha.fitness 存到内存，最后一次性写入文件
    double *alpha_history = (double *)malloc(g_max_iter * sizeof(double));
    if (!alpha_history) {
        fprintf(stderr, "Error: Memory allocation failed for alpha_history.\n");
        free(population);
        exit(EXIT_FAILURE);
    }

    // GWO 主循环
    for (int iter = 1; iter <= g_max_iter; iter++) {
        // 计算适应度
        for (int i = 0; i < g_pop_size; i++) {
            population[i].fitness = info->function(population[i].position, g_dimension);
        }

        // 排序并取出前三个
        sort_population(population, g_pop_size);
        Wolf alpha = population[0];
        Wolf beta  = population[1];
        Wolf delta = population[2];

        // 计算收缩参数 a
        double a = 2.0 - (2.0 * iter / g_max_iter);

        // 更新位置
        for (int i = 0; i < g_pop_size; i++) {
            for (int d = 0; d < g_dimension; d++) {
                // 针对 alpha
                double r1 = (double)rand() / RAND_MAX;
                double r2 = (double)rand() / RAND_MAX;
                double A1 = 2.0 * a * r1 - a;
                double C1 = 2.0 * r2;
                double D_alpha = fabs(C1 * alpha.position[d] - population[i].position[d]);
                double X1 = alpha.position[d] - A1 * D_alpha;

                // 针对 beta
                r1 = (double)rand() / RAND_MAX;
                r2 = (double)rand() / RAND_MAX;
                double A2 = 2.0 * a * r1 - a;
                double C2 = 2.0 * r2;
                double D_beta = fabs(C2 * beta.position[d] - population[i].position[d]);
                double X2 = beta.position[d] - A2 * D_beta;

                // 针对 delta
                r1 = (double)rand() / RAND_MAX;
                r2 = (double)rand() / RAND_MAX;
                double A3 = 2.0 * a * r1 - a;
                double C3 = 2.0 * r2;
                double D_delta = fabs(C3 * delta.position[d] - population[i].position[d]);
                double X3 = delta.position[d] - A3 * D_delta;

                // 最终更新
                population[i].position[d] = (X1 + X2 + X3) / 3.0;
            }
        }

        // 将本代 alpha 的适应度保存
        alpha_history[iter - 1] = alpha.fitness;
    }

    clock_t end = clock();
    double total_time = (double)(end - start) / CLOCKS_PER_SEC;

    // 计算纯算法时间
    double algorithm_time = total_time;

    // 最后一次性写出收敛数据
    FILE *fp_convergence = fopen(convergence_file, "w");
    if (fp_convergence) {
        for (int iter = 1; iter <= g_max_iter; iter++) {
            fprintf(fp_convergence, "%d %.6f\n", iter, alpha_history[iter - 1]);
        }
        fclose(fp_convergence);
    } else {
        fprintf(stderr, "Warning: Failed to open %s for writing convergence.\n", convergence_file);
    }

    // 写性能日志
    write_performance_log(performance_file, algorithm_time);

    // 释放内存
    free(alpha_history);
    free(population);

    return 0;
}
