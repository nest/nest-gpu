/*
	Matt Dean - https://github.com/mattdean1/cuda
*/

#ifndef SCANCUH
#define SCANCUH

__global__ void prescan_arbitrary(int *g_odata, int *g_idata, int n, int powerOfTwo);
__global__ void prescan_arbitrary_unoptimized(int *g_odata, int *g_idata, int n, int powerOfTwo);

__global__ void prescan_large(int *g_odata, int *g_idata, int n, int* sums);
__global__ void prescan_large_unoptimized(int *output, int *input, int n, int *sums);

__global__ void add(int *output, int length, int *n1);
__global__ void add(int *output, int length, int *n1, int *n2);

void _checkCudaError(const char *message, cudaError_t err, const char *caller);
void printResult(const char* prefix, int result, long nanoseconds);
void printResult(const char* prefix, int result, float milliseconds);

bool isPowerOfTwo(int x);
int nextPowerOfTwo(int x);

long get_nanos();

void prefix_scan(int *output, int *input, int length, bool bcao);
void scanLargeDeviceArray(int *output, int *input, int length, bool bcao);
void scanSmallDeviceArray(int *d_out, int *d_in, int length, bool bcao);
void scanLargeEvenDeviceArray(int *output, int *input, int length, bool bcao);

#endif
