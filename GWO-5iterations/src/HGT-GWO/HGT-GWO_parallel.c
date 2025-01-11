#include "../common/GWO.h"
#include <mpi.h>

// Historical position arrays
static double *previous_best_pos = NULL;
static double *previous_positions = NULL;

// Weights for HGT-GWO
double alpha_weight = 0.1;
double beta_weight  = 0.1;

// Get the top 3 wolves in the local population
static void get_local_top3(Wolf *local_pop, int local_size) {
    // Directly sort and take the top 3
    sort_population(local_pop, local_size);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 4) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <test_function_name> <dimension> <num_cores>\n", argv[0]);
        }
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    char *test_function_name = argv[1];
    g_dimension = atoi(argv[2]);
    int num_cores = atoi(argv[3]);

    if (g_dimension <= 0 || num_cores <= 0 || num_cores != size) {
        if (rank == 0) {
            fprintf(stderr, "Error: Invalid dimension or core count.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    // Retrieve test function information
    TestFunctionInfo *info = get_test_function_info(test_function_name);
    if (!info) {
        if (rank == 0) {
            fprintf(stderr, "Error: Unknown test function: %s\n", test_function_name);
        }
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    // Allocate historical arrays for HGT-GWO
    previous_best_pos  = (double*)malloc(g_dimension*sizeof(double));
    previous_positions = (double*)malloc(g_pop_size*g_dimension*sizeof(double));
    if(!previous_best_pos || !previous_positions){
        perror("Error allocating memory for previous_* arrays");
        MPI_Abort(MPI_COMM_WORLD,-1);
    }

    // Define MPI_WOLF to pass wolf position+fitness
    MPI_Datatype MPI_WOLF;
    {
        int block_lengths[2]      = {g_dimension, 1};
        MPI_Aint displacements[2] = {
            offsetof(Wolf, position),
            offsetof(Wolf, fitness)
        };
        MPI_Datatype types[2]     = {MPI_DOUBLE, MPI_DOUBLE};
        MPI_Type_create_struct(2, block_lengths, displacements, types, &MPI_WOLF);
        MPI_Type_commit(&MPI_WOLF);
    }

    // Local population
    int local_size = g_pop_size/size + ( (rank < (g_pop_size%size)) ? 1 : 0 );
    Wolf *local_pop = (Wolf*)malloc(local_size*sizeof(Wolf));
    if(!local_pop){
        perror("Error allocating local_pop");
        MPI_Abort(MPI_COMM_WORLD,-1);
    }

    Wolf alpha, beta, delta;
    double *alpha_history = NULL;

    // Logging related
    char core_info[32];
    snprintf(core_info,sizeof(core_info),"%d_cores",num_cores);
    char convergence_dir[256], performance_dir[256];
    char convergence_file[256], performance_file[256];

    // Only rank=0 has the global population and initializes it
    Wolf *population=NULL;
    if(rank==0){
        generate_directory_path(convergence_dir, sizeof(convergence_dir),
                                "/home/yuhang.jiang/Project/data/convergence",
                                test_function_name, g_dimension, core_info);

        generate_directory_path(performance_dir, sizeof(performance_dir),
                                "/home/yuhang.jiang/Project/data/performance_logs",
                                test_function_name, g_dimension, core_info);

        create_directory_recursively(convergence_dir);
        create_directory_recursively(performance_dir);

        snprintf(convergence_file,sizeof(convergence_file),
                 "%s/convergence_HGT-GWO_parallel.txt", convergence_dir);
        snprintf(performance_file,sizeof(performance_file),
                 "%s/performance_log_HGT-GWO_parallel.txt", performance_dir);

        population = (Wolf*)malloc(g_pop_size*sizeof(Wolf));
        if(!population){
            perror("Error allocating memory for population");
            MPI_Abort(MPI_COMM_WORLD,-1);
        }

        srand(12345);
        initialize_population(population, g_pop_size, g_dimension,
                              info->lower_bound, info->upper_bound);
        sort_population(population,g_pop_size);

        alpha = population[0];
        beta  = population[1];
        delta = population[2];

        // Record the previous best solution position
        memcpy(previous_best_pos, alpha.position, g_dimension*sizeof(double));
        // Record the initial positions of all individuals
        for(int i=0;i<g_pop_size;i++){
            memcpy(&previous_positions[i*g_dimension],
                   population[i].position, g_dimension*sizeof(double));
        }

        // Allocate alpha_history
        alpha_history = (double*)malloc(g_max_iter*sizeof(double));
        if(!alpha_history){
            fprintf(stderr,"Error: alpha_history.\n");
            MPI_Abort(MPI_COMM_WORLD,-1);
        }
    }

    // Scatterv distributes the initial population
    int *sendcounts = (int*)malloc(size*sizeof(int));
    int *displs     = (int*)malloc(size*sizeof(int));
    {
        int offset=0;
        for(int i=0;i<size;i++){
            int portion= g_pop_size/size + ( (i<(g_pop_size%size)) ? 1:0 );
            sendcounts[i]= portion*sizeof(Wolf);
            displs[i]    = offset;
            offset       += sendcounts[i];
        }
    }
    MPI_Scatterv(population, sendcounts, displs, MPI_BYTE,
                 local_pop, local_size*sizeof(Wolf),
                 MPI_BYTE, 0, MPI_COMM_WORLD);

    double start_time = MPI_Wtime();
    double io_time    = 0.0;


    int sync_interval = 5;

    Wolf local_top3[3];
    Wolf *global_top3 = NULL;
    if(rank==0){
        global_top3 = (Wolf*)malloc(3*size*sizeof(Wolf));
        if(!global_top3){
            fprintf(stderr,"Error allocating global_top3.\n");
            MPI_Abort(MPI_COMM_WORLD,-1);
        }
    }

    // Main loop
    for(int iter = 1; iter <= g_max_iter; iter++){
        // 1) Calculate local fitness
        for(int i=0; i<local_size; i++){
            local_pop[i].fitness = info->function(local_pop[i].position, g_dimension);
        }

        // 2) Check if global synchronization is needed
        if(iter % sync_interval == 0){
            // Select local top 3
            get_local_top3(local_pop, local_size);
            local_top3[0] = local_pop[0];
            local_top3[1] = local_pop[1];
            local_top3[2] = local_pop[2];

            // Gather to rank=0
            MPI_Gather(local_top3, 3, MPI_WOLF,
                       global_top3, 3, MPI_WOLF,
                       0, MPI_COMM_WORLD);

            if(rank==0){
                // Sort 3*size to get global alpha, beta, delta
                sort_population(global_top3, 3*size);
                alpha = global_top3[0];
                beta  = global_top3[1];
                delta = global_top3[2];

                // Record convergence information
                alpha_history[iter-1] = alpha.fitness;
            }

            // Broadcast alpha, beta, delta to all processes
            MPI_Bcast(&alpha, 1, MPI_WOLF, 0, MPI_COMM_WORLD);
            MPI_Bcast(&beta,  1, MPI_WOLF, 0, MPI_COMM_WORLD);
            MPI_Bcast(&delta, 1, MPI_WOLF, 0, MPI_COMM_WORLD);

            // 3) Update positions locally (HGT-GWO logic)
            double a = 2.0 - 2.0*((double)iter/g_max_iter);
            for(int i=0; i<local_size;i++){
                for(int d=0; d<g_dimension; d++){
                    // Standard GWO alpha part (beta, delta can be added as needed)
                    double r1=(double)rand()/RAND_MAX;
                    double r2=(double)rand()/RAND_MAX;
                    double A1=2.0*a*r1 - a;
                    double C1=2.0*r2;
                    double D_alpha = fabs(C1*alpha.position[d] - local_pop[i].position[d]);
                    double X1= alpha.position[d] - A1*D_alpha;

                    // HGT: Historical guidance + trend term
                    double hist_guidance= previous_best_pos[d];
                    int idx = (i + displs[rank]/sizeof(Wolf))*g_dimension + d;
                    double trend_adjust= local_pop[i].position[d] - previous_positions[idx];

                    // Merge
                    local_pop[i].position[d] =
                        (1.0 - alpha_weight - beta_weight)*X1
                        + alpha_weight*hist_guidance
                        + beta_weight*trend_adjust;
                }
            }

            if(rank==0){
                // Update previous_best_pos
                memcpy(previous_best_pos, alpha.position, g_dimension*sizeof(double));
                // Synchronize population[...] -> previous_positions[...]
                for(int i=0; i<g_pop_size; i++){
                    // population[i].position => previous_positions[i*g_dimension...]
                    memcpy(&previous_positions[i*g_dimension],
                           population[i].position, g_dimension*sizeof(double));
                }
            }
        }
        else {
            // Non-synchronization rounds
            if(rank==0){
                // Only record alpha.fitness
                alpha_history[iter-1] = alpha.fitness;
            }
            // Perform local updates with the latest alpha, beta, delta
            double a= 2.0 - 2.0*((double)iter/g_max_iter);
            for(int i=0;i<local_size;i++){
                for(int d=0; d<g_dimension; d++){
                    double r1=(double)rand()/RAND_MAX;
                    double r2=(double)rand()/RAND_MAX;
                    double A1=2.0*a*r1 - a;
                    double C1=2.0*r2;
                    double D_alpha= fabs(C1*alpha.position[d] - local_pop[i].position[d]);
                    double X1= alpha.position[d] - A1*D_alpha;

                    double hist_guidance= previous_best_pos[d];
                    int idx = (i + displs[rank]/sizeof(Wolf))*g_dimension + d;
                    double trend_adjust= local_pop[i].position[d] - previous_positions[idx];

                    local_pop[i].position[d] =
                        (1.0 - alpha_weight - beta_weight)*X1
                        + alpha_weight*hist_guidance
                        + beta_weight*trend_adjust;
                }
            }
        }

        // 4) Update local previous_positions every round
        //    so it is available for the next round
        for(int i=0;i<local_size;i++){
            for(int d=0; d<g_dimension; d++){
                previous_positions[(i+displs[rank]/sizeof(Wolf))*g_dimension + d]
                    = local_pop[i].position[d];
            }
        }
    }

    double end_time = MPI_Wtime();

    // rank=0 writes logs
    if(rank==0){
        double total_time= end_time - start_time;
        double algorithm_time= total_time - io_time;

        // Write convergence to file
        FILE *fp_conv= fopen(convergence_file,"w");
        if(fp_conv){
            for(int i=0; i< g_max_iter; i++){
                fprintf(fp_conv, "%d %.6f\n", i+1, alpha_history[i]);
            }
            fclose(fp_conv);
        } else {
            fprintf(stderr,"Warning: failed to open %s\n", convergence_file);
        }

        // Write performance log
        write_performance_log(performance_file, algorithm_time);

        free(alpha_history);
        free(population);
        free(global_top3);
    }

    free(local_pop);
    free(sendcounts);
    free(displs);

    free(previous_best_pos);
    free(previous_positions);

    MPI_Type_free(&MPI_WOLF);
    MPI_Finalize();
    return 0;
}