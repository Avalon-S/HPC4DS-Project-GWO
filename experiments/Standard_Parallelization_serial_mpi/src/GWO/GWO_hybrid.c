#include "../common/GWO.h"
#include <mpi.h>
#include <omp.h>

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

    // Logging
    char core_info[32];
    snprintf(core_info, sizeof(core_info), "%d_cores", num_cores);

    char convergence_dir[256], performance_dir[256];
    char convergence_file[256], performance_file[256];

    if (rank == 0) {
        generate_directory_path(convergence_dir, sizeof(convergence_dir),
                                "/home/yuhang.jiang/Project/data/convergence",
                                test_function_name, g_dimension, core_info);

        generate_directory_path(performance_dir, sizeof(performance_dir),
                                "/home/yuhang.jiang/Project/data/performance_logs",
                                test_function_name, g_dimension, core_info);

        create_directory_recursively(convergence_dir);
        create_directory_recursively(performance_dir);

        snprintf(convergence_file, sizeof(convergence_file),
                 "%s/convergence_GWO_hybrid.txt", convergence_dir);
        snprintf(performance_file, sizeof(performance_file),
                 "%s/performance_log_GWO_hybrid.txt", performance_dir);
    }

    // Define MPI_WOLF
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

    // local_size
    int local_size = g_pop_size / size + ((rank < (g_pop_size % size)) ? 1 : 0);
    Wolf *local_pop = (Wolf *)malloc(local_size*sizeof(Wolf));
    if(!local_pop){
        perror("Error allocating memory for local population");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    // rank=0 has full population
    Wolf alpha, beta, delta;
    Wolf *population = NULL;

    if(rank==0){
        srand(12345);
        population = (Wolf*)malloc(g_pop_size*sizeof(Wolf));
        if(!population){
            perror("Error allocating memory for population");
            MPI_Abort(MPI_COMM_WORLD,-1);
        }

        initialize_population(population, g_pop_size, g_dimension,
                             info->lower_bound, info->upper_bound);
        sort_population(population, g_pop_size);

        alpha = population[0];
        beta  = population[1];
        delta = population[2];
    }

    // Prepare scatter
    int *sendcounts=(int*)malloc(size*sizeof(int));
    int *displs=(int*)malloc(size*sizeof(int));
    {
        int offset=0;
        for(int i=0;i<size;i++){
            int portion= g_pop_size/size + ((i<(g_pop_size%size))?1:0);
            sendcounts[i]= portion*sizeof(Wolf);
            displs[i]    = offset;
            offset      += sendcounts[i];
        }
    }

    // First scatter
    MPI_Scatterv(population, sendcounts, displs, MPI_BYTE,
                 local_pop, local_size*sizeof(Wolf),
                 MPI_BYTE, 0, MPI_COMM_WORLD);

    double start_time=MPI_Wtime();
    double io_time=0.0; // for I/O time

    int sync_interval=1;

    for(int iter=1; iter<= g_max_iter; iter++){
        // 1) each process parallel fitness
        #pragma omp parallel for
        for(int i=0;i<local_size;i++){
            local_pop[i].fitness= info->function(local_pop[i].position, g_dimension);
        }

        if(iter % sync_interval==0){
            // gather
            MPI_Gatherv(local_pop, local_size*sizeof(Wolf), MPI_BYTE,
                        population, sendcounts, displs, MPI_BYTE,
                        0, MPI_COMM_WORLD);

            if(rank==0){
                sort_population(population,g_pop_size);
                alpha = population[0];
                beta  = population[1];
                delta = population[2];

                // parallel update position
                #pragma omp parallel for
                for(int i=0; i< g_pop_size; i++){
                    for(int d=0; d< g_dimension; d++){
                        double r1=(double)rand()/RAND_MAX;
                        double r2=(double)rand()/RAND_MAX;
                        double a= 2.0 - (2.0*(double)iter/g_max_iter);

                        double A1= 2.0*a*r1 - a;
                        double C1= 2.0*r2;
                        double D_alpha= fabs(C1*alpha.position[d] - population[i].position[d]);
                        double X1= alpha.position[d] - A1*D_alpha;

                        r1=(double)rand()/RAND_MAX;
                        r2=(double)rand()/RAND_MAX;
                        double A2= 2.0*a*r1 - a;
                        double C2= 2.0*r2;
                        double D_beta= fabs(C2*beta.position[d] - population[i].position[d]);
                        double X2= beta.position[d] - A2*D_beta;

                        r1=(double)rand()/RAND_MAX;
                        r2=(double)rand()/RAND_MAX;
                        double A3= 2.0*a*r1 - a;
                        double C3= 2.0*r2;
                        double D_delta= fabs(C3*delta.position[d] - population[i].position[d]);
                        double X3= delta.position[d] - A3*D_delta;

                        population[i].position[d] = (X1+X2+X3)/3.0;
                    }
                }

                double io_start=MPI_Wtime();
                write_convergence_to_file(convergence_file, iter, alpha.fitness);
                double io_end=MPI_Wtime();
                io_time += (io_end - io_start);
            }

            // broadcast alpha,beta,delta
            MPI_Bcast(&alpha,1,MPI_WOLF,0,MPI_COMM_WORLD);
            MPI_Bcast(&beta, 1,MPI_WOLF,0,MPI_COMM_WORLD);
            MPI_Bcast(&delta,1,MPI_WOLF,0,MPI_COMM_WORLD);

            // scatter updated population
            MPI_Scatterv(population, sendcounts, displs, MPI_BYTE,
                         local_pop, local_size*sizeof(Wolf),
                         MPI_BYTE,0,MPI_COMM_WORLD);
        } else {
            // No global sync
            if(rank==0){
                // Just record alpha fitness in the iteration's output
                write_convergence_to_file(convergence_file, iter, alpha.fitness);
            }
        }
    }

    double end_time=MPI_Wtime();

    if(rank==0){
        // double algorithm_time= (end_time - start_time) - io_time;
        double algorithm_time= (end_time - start_time) - io_time;
        write_performance_log(performance_file, algorithm_time);
    }

    free(local_pop);
    if(rank==0){
        free(population);
    }
    free(sendcounts);
    free(displs);

    MPI_Type_free(&MPI_WOLF);
    MPI_Finalize();
    return 0;
}
