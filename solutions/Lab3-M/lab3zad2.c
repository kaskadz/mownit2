#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <sys/resource.h>
#include <gsl/gsl_blas.h>
#include <string.h>

//==> globals
int MEASUREMENTS_COUNT;
int REPEAT_COUNT = 10;
//<== end globals

//==> time measurement utilities
struct se_time
{
    struct timeval start;
    struct timeval end;
};

struct times_struct
{
    struct se_time user;
    struct se_time sys;
    struct se_time real;
    struct rusage usage;
};

struct times_struct time_measurement;
double real_time;

void record_start_time()
{
    printf("--> Start\n");
    getrusage(RUSAGE_SELF, &time_measurement.usage);
    time_measurement.real.start = time_measurement.usage.ru_stime;
    time_measurement.user.start = time_measurement.usage.ru_utime;
    gettimeofday(&time_measurement.real.start, NULL);
}

void record_end_time()
{
    printf("<-- Stop\n");
    getrusage(RUSAGE_SELF, &time_measurement.usage);
    time_measurement.real.end = time_measurement.usage.ru_stime;
    time_measurement.user.end = time_measurement.usage.ru_utime;
    gettimeofday(&time_measurement.real.end, NULL);

    real_time = ((double)(time_measurement.real.end.tv_sec - time_measurement.real.start.tv_sec) + (double)((time_measurement.real.end.tv_usec) - time_measurement.real.start.tv_usec) * pow(10, -6));
}
//<== end time measurement utilities

//==> matrics/vector generation
double *generate_random_vector(int vector_size)
{

    double *vector = malloc(vector_size * sizeof(double));
    for (int i = 0; i < vector_size; i++)
    {
        vector[i] = (double)rand() / (double)rand();
    }
    return vector;
}

double **generate_random_matrix(int size_x, int size_y)
{
    double **A = malloc(size_y * sizeof(double *));
    for (int i = 0; i < size_x; i++)
    {
        A[i] = malloc(size_y * sizeof(double));
    }

    for (int i = 0; i < size_x; i++)
    {
        for (int j = 0; j < size_y; j++)
        {
            A[i][j] = (double)rand();
        }
    }

    return A;
}

double **generate_zeros_matrix(int size_x, int size_y)
{
    double **A = malloc(size_y * sizeof(double *));
    for (int i = 0; i < size_x; i++)
    {
        A[i] = malloc(size_y * sizeof(double));
    }

    for (int i = 0; i < size_x; i++)
    {
        for (int j = 0; j < size_y; j++)
        {
            A[i][j] = 0.0;
        }
    }

    return A;
}

//==> multiplication
double naive_mul(int size)
{

    double **A = generate_random_matrix(size, size);
    double **B = generate_random_matrix(size, size);
    double **C = generate_zeros_matrix(size, size);

    record_start_time();

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            for (int k = 0; k < size; k++)
            {
                C[j][i] = C[j][i] + A[k][i] * B[j][k];
            }
        }
    }

    record_end_time();

    free(A);
    free(B);
    free(C);

    return real_time;
}

double better_mul(int size)
{

    double **A = generate_random_matrix(size, size);
    double **B = generate_random_matrix(size, size);
    double **C = generate_zeros_matrix(size, size);

    record_start_time();

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            for (int k = 0; k < size; k++)
            {
                C[i][k] = C[i][k] + A[j][k] * B[i][j];
            }
        }
    }

    record_end_time();

    free(A);
    free(B);
    free(C);

    return real_time;
}

double blas_mul(int size)
{

    double *a = malloc(size * size * sizeof(double));
    double *b = malloc(size * size * sizeof(double));
    double *c = malloc(size * size * sizeof(double));

    a = generate_random_vector(size * size);
    b = generate_random_vector(size * size);

    gsl_matrix_view X = gsl_matrix_view_array(a, size, size);
    gsl_matrix_view Y = gsl_matrix_view_array(b, size, size);
    gsl_matrix_view Z = gsl_matrix_view_array(c, size, size);

    record_start_time();

    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &X.matrix, &Y.matrix, 0.0, &Z.matrix);

    record_end_time();

    free(a);
    free(b);
    free(c);

    return real_time;
}
//<== end multiplication

//==> saving results
void generate_output_csv_file(int *sizes, double *naive_results, double *better_results, double *blas_results)
{
    FILE *file_pointer;
    file_pointer = fopen("results.csv", "w");

    // header
    fprintf(file_pointer, "type,size,time\n");

    // records
    for (int i = 0; i < MEASUREMENTS_COUNT; i++)
    {
        for (int j = 0; j < REPEAT_COUNT; j++)
        {
            fprintf(file_pointer, "naive,%d,%f\n", sizes[i], naive_results[i * REPEAT_COUNT + j]);
        }
    }

    for (int i = 0; i < MEASUREMENTS_COUNT; i++)
    {
        for (int j = 0; j < REPEAT_COUNT; j++)
        {
            fprintf(file_pointer, "better,%d,%f\n", sizes[i], better_results[i * REPEAT_COUNT + j]);
        }
    }

    for (int i = 0; i < MEASUREMENTS_COUNT; i++)
    {
        for (int j = 0; j < REPEAT_COUNT; j++)
        {
            fprintf(file_pointer, "blas,%d,%f\n", sizes[i], blas_results[i * REPEAT_COUNT + j]);
        }
    }

    fclose(file_pointer);
}
//<== end saving results

//==> main
int main(int argc, char **argv)
{
    int sizes[] = {1, 50, 100, 500, 1000, 1100, 1200, 1300, 1400, 1500};

    MEASUREMENTS_COUNT = sizeof(sizes) / sizeof(int); // number of "size" parameters

    double *naive_results = malloc(MEASUREMENTS_COUNT * REPEAT_COUNT * sizeof(double));
    double *better_results = malloc(MEASUREMENTS_COUNT * REPEAT_COUNT * sizeof(double));
    double *blas_results = malloc(MEASUREMENTS_COUNT * REPEAT_COUNT * sizeof(double));

    for (int i = 0; i < MEASUREMENTS_COUNT; i++)
    {
        for (int j = 0; j < REPEAT_COUNT; j++)
        {
            printf("%d: \n", i * REPEAT_COUNT + j);
            naive_results[i * REPEAT_COUNT + j] = naive_mul(sizes[i]);
            better_results[i * REPEAT_COUNT + j] = better_mul(sizes[i]);
            blas_results[i * REPEAT_COUNT + j] = blas_mul(sizes[i]);
        }
    }

    generate_output_csv_file(sizes, naive_results, better_results, blas_results);

    free(naive_results);
    free(better_results);
    free(blas_results);

    return 0;
}
//<== end main
