#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <omp.h>

void fillMatrix(int N, double** A, double* b);
void partialPivot(int N, double** A, double* b, int currentPosition);
void printMatrix(int N, double** A, double* b);
void backSubstitution(int N, double** A, double* b, double* x, int numThreads);
void printSolutionVector(double* x, int N);

int main(int argc, char** argv){

	int N = atoi(argv[1]);
	int numThreads = atoi(argv[2]);
	pthread_t elimination_threads[numThreads];

	// Ax = b

	//Allocate Matrix 'A'
	double **A = (double **)calloc(N,sizeof(double*));
	for (int q=0; q < N; q++)
		A[q] = (double*)calloc(N,sizeof(double*));

	//Allocate Vector 'b'
	double* b = (double*) malloc(sizeof(double)*N);
	double* x = (double*) malloc(sizeof(double)*N);

	fillMatrix(N, A, b); //random 0-1000

	if (N <= 8)	
	printMatrix(N, A, b);

	int i, k;
	double m;

	//Gaussian Elimination
	omp_set_nested(1);
	for (int j=0; j < N-1; j++){
		partialPivot(N, A, b, j);

		#pragma omp parallel default(none) num_threads(numThreads) shared(N,A,b,j) private(i,k,m)
		#pragma omp for schedule(static)
			
		for (int k=j+1; k<N; k++){
			m = A[k][j]/A[j][j];
			for (i=j; i<N; i++){
				A[k][i] = A[k][i] - (m * A[j][i]);
			}
			b[k] = b[k] - (m * b[j]);
		}

	}
	if (N <= 8){
		printf("\nBack Sustitution matriz (A) y el vector (b):\n\n");
		printMatrix(N,A,b);
	}

	backSubstitution(N, A, b, x, numThreads);
	
	printSolutionVector(x, N);
	
}

void partialPivot(int n, double** a, double* b, int j){

	int   i,k,m,rowx;
	double xfac, temp, temp1, amax;

	amax = (double) fabs(a[j][j]) ;
    m = j;
    for (i=j+1; i<n; i++){   /* Encuentra la fila con el pivote más grande */
    	xfac = (double) fabs(a[i][j]);
    	if(xfac > amax) {amax = xfac; m=i;}
    }

    if(m != j) {  /* Intercambio de filas */
    	rowx = rowx+1;
    	temp1 = b[j];
    	b[j]  = b[m];
    	b[m]  = temp1;
    	for(k=j; k<n; k++) {
    		temp = a[j][k];
    		a[j][k] = a[m][k];
    		a[m][k] = temp;
    	}
    }
}

void fillMatrix(int N, double** A, double* b){
	int i, j;
	for (i=0; i<N; i++){
		for (j=0; j<N; j++){
			A[i][j] = (drand48()*1000);
		}
		b[i] = (drand48()*1000);
	}
}

void printMatrix(int N, double** A, double* b){
	if (N <= 8){
		for (int x=0; x<N; x++){
			printf("| ");
			for(int y=0; y<N; y++)
				printf("%7.2f ", A[x][y]);
			printf("| | %7.2f |\n", b[x]);
		}
	}
	else{
		printf("\nMatriz y vector demasiado grande para imprimir.\n");
		printf("Si desea ver la salida, intente de nuevo con una matriz de longitud 8 o menor.\n");
	}
}

void printSolutionVector(double* x, int N){
	if (N <= 8){
		printf("\nVector Solucion (x):\n\n");
		for (int i=0; i<N; i++){
			printf("|%10.6f|\n", x[i]);
		}
	}
}

void backSubstitution(int N, double** A, double* b, double* x, int numThreads){
	int i,j;
	for (i=N-1; i >= 0; i--){
		x[i] = b[i];
		for (j=i+1; j<N; j++){
			x[i] -= A[i][j]*x[j];
		}
		x[i] = x[i] / A[i][i];
	}
}



