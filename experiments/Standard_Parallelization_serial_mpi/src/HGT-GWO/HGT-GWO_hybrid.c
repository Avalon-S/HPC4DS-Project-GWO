#include "../common/GWO.h"
#include <mpi.h>
#include <omp.h>

// Historical arrays
static double *previous_best_pos = NULL;
static double *previous_positions = NULL;

// Weights for HGT-GWO
double alpha_weight = 0.1;
double beta_weight  = 0.1;

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(argc<4){
        if(rank==0){
            fprintf(stderr,"Usage: %s <test_function_name> <dimension> <num_cores>\n",argv[0]);
        }
        MPI_Abort(MPI_COMM_WORLD,-1);
    }

    char *test_function_name= argv[1];
    g_dimension= atoi(argv[2]);
    int num_cores= atoi(argv[3]);

    if(g_dimension<=0 || num_cores<=0 || num_cores!=size){
        if(rank==0){
            fprintf(stderr,"Error: Invalid dimension or core count.\n");
        }
        MPI_Abort(MPI_COMM_WORLD,-1);
    }

    // e.g. g_pop_size=200; g_max_iter=500; or use default in GWO.h

    // get test function info
    TestFunctionInfo *info= get_test_function_info(test_function_name);
    if(!info){
        if(rank==0){
            fprintf(stderr,"Error: Unknown test function: %s\n", test_function_name);
        }
        MPI_Abort(MPI_COMM_WORLD,-1);
    }

    // allocate historical arrays
    previous_best_pos= (double*)malloc(g_dimension*sizeof(double));
    previous_positions= (double*)malloc(g_pop_size*g_dimension*sizeof(double));
    if(!previous_best_pos || !previous_positions){
        perror("Error allocating memory for previous_* arrays");
        MPI_Abort(MPI_COMM_WORLD,-1);
    }

    // directories
    char core_info[32];
    snprintf(core_info,sizeof(core_info),"%d_cores",num_cores);

    char convergence_dir[256], performance_dir[256];
    char convergence_file[256], performance_file[256];

    double *alpha_history= NULL;

    if(rank==0){
        generate_directory_path(convergence_dir,sizeof(convergence_dir),
                                "/home/yuhang.jiang/Project/data/convergence",
                                test_function_name, g_dimension, core_info);

        generate_directory_path(performance_dir,sizeof(performance_dir),
                                "/home/yuhang.jiang/Project/data/performance_logs",
                                test_function_name, g_dimension, core_info);

        create_directory_recursively(convergence_dir);
        create_directory_recursively(performance_dir);

        snprintf(convergence_file,sizeof(convergence_file),
                 "%s/convergence_HGT-GWO_hybrid.txt",convergence_dir);
        snprintf(performance_file,sizeof(performance_file),
                 "%s/performance_log_HGT-GWO_hybrid.txt",performance_dir);
    }

    // define MPI_WOLF
    MPI_Datatype MPI_WOLF;
    {
        int block_lengths[2]= {g_dimension,1};
        MPI_Aint displacements[2]={
            offsetof(Wolf, position),
            offsetof(Wolf, fitness)
        };
        MPI_Datatype types[2]= {MPI_DOUBLE,MPI_DOUBLE};
        MPI_Type_create_struct(2, block_lengths, displacements, types, &MPI_WOLF);
        MPI_Type_commit(&MPI_WOLF);
    }

    int local_size= g_pop_size/size + (rank< g_pop_size%size?1:0);
    Wolf* local_pop= (Wolf*)malloc(local_size*sizeof(Wolf));
    if(!local_pop){
        perror("Error allocating local population");
        MPI_Abort(MPI_COMM_WORLD,-1);
    }

    Wolf alpha,beta,delta;
    Wolf* population=NULL;
    double io_time=0.0;

    if(rank==0){
        population= (Wolf*)malloc(g_pop_size*sizeof(Wolf));
        if(!population){
            perror("Error allocating memory for population");
            MPI_Abort(MPI_COMM_WORLD,-1);
        }

        srand(12345);
        initialize_population(population,g_pop_size,g_dimension,
                              info->lower_bound, info->upper_bound);
        sort_population(population,g_pop_size);
        alpha= population[0];
        beta = population[1];
        delta= population[2];

        memcpy(previous_best_pos, alpha.position, g_dimension*sizeof(double));
        for(int i=0;i<g_pop_size;i++){
            memcpy(&previous_positions[i*g_dimension],
                   population[i].position,g_dimension*sizeof(double));
        }

        alpha_history= (double*)malloc(g_max_iter*sizeof(double));
        if(!alpha_history){
            fprintf(stderr,"Error allocating alpha_history.\n");
            MPI_Abort(MPI_COMM_WORLD,-1);
        }
    }

    // distribute initial population
    int *sendcounts= (int*)malloc(size*sizeof(int));
    int *displs= (int*)malloc(size*sizeof(int));
    {
        int offset=0;
        for(int i=0;i<size;i++){
            int portion= g_pop_size/size + (i< g_pop_size%size?1:0);
            sendcounts[i]= portion*sizeof(Wolf);
            displs[i]    = offset;
            offset      += sendcounts[i];
        }
    }

    MPI_Scatterv(population, sendcounts, displs, MPI_BYTE,
                 local_pop, local_size*sizeof(Wolf),
                 MPI_BYTE,0,MPI_COMM_WORLD);

    double start_time= MPI_Wtime();

    // sync_interval=2
    int sync_interval=1;

    // main GWO loop
    for(int iter=1; iter<= g_max_iter; iter++){
        // 1) local fitness
        #pragma omp parallel for
        for(int i=0;i< local_size; i++){
            local_pop[i].fitness= info->function(local_pop[i].position, g_dimension);
        }

        // only gather+sort+scatter every sync_interval generations
        if(iter % sync_interval==0){
            // gather
            MPI_Gatherv(local_pop, local_size*sizeof(Wolf), MPI_BYTE,
                        population, sendcounts, displs, MPI_BYTE,
                        0, MPI_COMM_WORLD);

            if(rank==0){
                sort_population(population,g_pop_size);
                alpha= population[0];
                beta = population[1];
                delta= population[2];

                double a= 2.0 - 2.0*((double)iter/g_max_iter);
                // parallel update (HGT-GWO logic)
                #pragma omp parallel for
                for(int i=0;i< g_pop_size; i++){
                    for(int d=0; d<g_dimension; d++){
                        double r1= (double)rand()/RAND_MAX;
                        double r2= (double)rand()/RAND_MAX;
                        double A1= 2.0*a*r1 - a;
                        double C1= 2.0*r2;
                        double D_alpha= fabs(C1*alpha.position[d] - population[i].position[d]);
                        double X1= alpha.position[d] - A1*D_alpha;

                        double hist_guidance= previous_best_pos[d];
                        double trend_adjust= population[i].position[d]
                                             - previous_positions[i*g_dimension + d];

                        population[i].position[d]=
                            (1.0 - alpha_weight - beta_weight)*X1
                            + alpha_weight*hist_guidance
                            + beta_weight*trend_adjust;
                    }
                }

                // update historical
                memcpy(previous_best_pos, alpha.position, g_dimension*sizeof(double));
                for(int i=0;i<g_pop_size;i++){
                    memcpy(&previous_positions[i*g_dimension],
                           population[i].position,g_dimension*sizeof(double));
                }
                alpha_history[iter-1]= alpha.fitness;
            }

            // broadcast
            MPI_Bcast(previous_best_pos, g_dimension, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(previous_positions, g_pop_size*g_dimension, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            // scatter updated population
            MPI_Scatterv(population, sendcounts, displs, MPI_BYTE,
                         local_pop, local_size*sizeof(Wolf),
                         MPI_BYTE, 0, MPI_COMM_WORLD);
        } else {
            // Not a sync iteration
            if(rank==0){
                alpha_history[iter-1] = alpha.fitness;
            }
        }
    }

    double end_time= MPI_Wtime();

    if(rank==0){
        double total_time= end_time - start_time;
        double algorithm_time= total_time - io_time;

        double io_begin= MPI_Wtime();
        FILE* fp_conv= fopen(convergence_file,"w");
        if(fp_conv){
            for(int i=0;i< g_max_iter;i++){
                fprintf(fp_conv, "%d %.6f\n", i+1, alpha_history[i]);
            }
            fclose(fp_conv);
        } else {
            fprintf(stderr,"Warning: fail to open %s\n", convergence_file);
        }
        double io_end= MPI_Wtime();
        io_time += (io_end - io_begin);

        double final_algo_time= total_time;// - io_time;
        write_performance_log(performance_file, final_algo_time);

        free(alpha_history);
        free(population);
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
