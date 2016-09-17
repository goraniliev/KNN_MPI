#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>

#define send_data_tag 2001
#define return_data_tag 2002
#define send_output_tag 2003
#define return_dist_tag 2004
#define return_cnt_tag 2005

#define DATA_SRC "iris.txt"
#define MAX_LINE_LEN 256
#define MAX_ROWS 150
#define COLUMNS 4
#define MAX_LABEL_LEN 25
#define NUM_ELEMENTS 100
#define MAX_PROCESSES 100

double inputData[MAX_ROWS][COLUMNS];
char outputData[MAX_ROWS][MAX_LABEL_LEN];
double recvData[MAX_ROWS][COLUMNS];
int labelsSent[MAX_ROWS], labelsRcvd[MAX_ROWS];
char recvLabels[MAX_PROCESSES][MAX_LABEL_LEN];
double partialDist[MAX_PROCESSES][3], partialCount[MAX_PROCESSES][3];
double distances[3], counts[3];

char *labels[] = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"};

int labelToId(char c[]) {
    if (strcmp(c, "Iris-setosa") == 0) {
        return 0;
    }
    if (strcmp(c, "Iris-virginica") == 0) {
        return 2;
    }
    return 1;
}

char *idToLabel(int id) {
    return labels[id];
}

void loadData() {
    FILE *fp = fopen(DATA_SRC, "r");
    char line[MAX_LINE_LEN];
    int row = 0;
    while (fgets(line, MAX_LINE_LEN, fp) != NULL) {
        int i;
        strtok(line, "\n"); // remove newline
        char tmp[MAX_LABEL_LEN];
        int c = 0, j = 0;
        for (i = 0; i < strlen(line); i++) {
            if (line[i] == ',') {
                inputData[row][c++] = strtod(tmp, NULL);
                memset(tmp, 0, MAX_LABEL_LEN);
                j = 0;
            }
            else {
                tmp[j++] = line[i];
            }
        }
        strcpy(outputData[row], tmp);
        labelsSent[row] = labelToId(tmp);
        row++;
    }
    fclose(fp);

}

double dist(double a[], double b[]) {
    double d = 0.0;
    int i;
    for (i = 0; i < COLUMNS; i++) {
        d += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return d;
}

double avg(double *a, int len) {
    int i;
    double s = 0.0;
    for (i = 0; i < len; i++) {
        s += a[i];
    }
    return 1.0 * s / len;
}

int main() {
    MPI_Init(NULL, NULL);
    MPI_Status status;
    int num_rows_to_receive;
    int i = 0;
    int world_size;
    //double distances[3], counts[3];
    //double partialDist[3], partialCount[3];
    double input[] = {5.0, 2.0, 3.5, 1.0};

    int ierr;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int avg_rows_per_process = MAX_ROWS / world_size;
    int num_rows_received;
    int num_rows_to_send;
    int sender;
    if (world_rank == 0) {
        loadData();
        int id_process;
        // split data across processes
        for (id_process = 1; id_process < world_size; id_process++) {
            int start_row = id_process * avg_rows_per_process + 1;
            int end_row = (id_process + 1) * avg_rows_per_process;
            if (MAX_ROWS - end_row < avg_rows_per_process) {
                end_row = MAX_ROWS - 1;
            }
            int num_rows_to_send = end_row - start_row + 1;
            printf("Process %d start=%d end = %d\n", id_process, start_row, end_row);
            ierr = MPI_Send(&num_rows_to_send, 1, MPI_INT, id_process, send_data_tag, MPI_COMM_WORLD);
            ierr = MPI_Send(&inputData[start_row], num_rows_to_send, MPI_DOUBLE, id_process, send_data_tag,
                            MPI_COMM_WORLD);

            ierr = MPI_Send(&labelsSent[start_row], num_rows_to_send, MPI_INT, id_process, send_output_tag,
                            MPI_COMM_WORLD);
        }

        double distances[3], counts[3];

        for (i = 0; i < avg_rows_per_process + 1; i++) {
            //      int id_label = labelToId(outputData[i]);
            int id_label = labelsSent[i];
            distances[id_label] += dist(input, inputData[i]);
            counts[id_label]++;
        }
        printf("Till partials good\n");
        // Add the partials from the other processes
        for (id_process = 1; id_process < world_size; id_process++) {
            ierr = MPI_Recv(&partialCount[id_process], 3, MPI_DOUBLE, MPI_ANY_SOURCE, return_cnt_tag, MPI_COMM_WORLD,
                            &status);
            ierr = MPI_Recv(&partialDist[id_process], 3, MPI_DOUBLE, MPI_ANY_SOURCE, return_dist_tag, MPI_COMM_WORLD,
                            &status);
            sender = status.MPI_SOURCE;
            for (i = 0; i < 3; i++) {
                distances[i] += partialDist[id_process][i];
                counts[i] += partialCount[id_process][i];
                printf("Process %d class %d Partial Count %f and Partial Dist %f \n", sender, i,
                       partialCount[id_process][i], partialDist[id_process][i]);
            }
        }
        printf("Before avg good\n");
        double minDistId = 0, minDist = distances[0] / counts[0];
        for (i = 0; i < 3; i++) {
            double avg = distances[i] / counts[i];
            if (avg < minDist) {
                minDist = avg;
                minDistId = i;
            }
            printf("Distance from class %d is %f", i, avg);
        }
        printf("The input belongs to class %s\n", idToLabel(minDistId));
    }
    else {//other process
        ierr = MPI_Recv(&num_rows_to_receive, 1, MPI_INT, 0, send_data_tag, MPI_COMM_WORLD, &status);
        ierr = MPI_Recv(&recvData[world_rank], num_rows_to_receive, MPI_DOUBLE, 0, send_data_tag, MPI_COMM_WORLD,
                        &status);
        ierr = MPI_Recv(&labelsRcvd, num_rows_to_receive, MPI_INT, 0, send_output_tag, MPI_COMM_WORLD, &status);

        num_rows_received = num_rows_to_receive;
        printf("Process %d processes %d rows\n", world_rank, num_rows_to_receive);
        for (i = 0; i < 3; i++) {
            printf("%d %f %s\n", world_rank, recvData[world_rank][i], recvLabels[i]);
        }

        //double partialDist[3], partialCount[3];
        double this_process_count[3], this_process_dist[3];
        for (i = 0; i < 3; i++) {
            partialCount[world_rank][i] = partialDist[world_rank][i] = 0;
        }
        for (i = 0; i < num_rows_received; i++) {
            int id_lbl = labelsRcvd[i];
            partialDist[world_rank][id_lbl] += dist(input, recvData[i]);
            partialCount[world_rank][id_lbl]++;
        }
        ierr = MPI_Send(&partialCount[world_rank], 3, MPI_DOUBLE, 0, return_cnt_tag, MPI_COMM_WORLD);
        ierr = MPI_Send(&partialDist[world_rank], 3, MPI_DOUBLE, 0, return_dist_tag, MPI_COMM_WORLD);
    }

    ierr = MPI_Finalize();
}

