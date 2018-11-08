#include "utilities.h"
#include <vector>

using namespace std;

typedef unsigned long long ULL;

const int MAX_DIMENSIONS = 10;

template <typename T>
void printArray(T* ar, int size) {
	cout << "Printing array:" << endl;
	for (int i = 0; i < size; i++) {
		cout << i << " :: " << ar[i] << endl;
	}
	cout << endl;
}

void printTestResults(long nDimensions, double radius, long totalPoints) {
	print("\nResults:");
	print("nDimensions: %ld", nDimensions);
	print("radius: %.2f", radius);
	print("total points: %ld\n", totalPoints);
}

void printCSVTestResults(int tpb, int bpg, int bpg1, int bpg2, long nDimensions, double radius, ULL totalPoints, float time) {
	debug("\nCSV Results:");
	debug("TPB, BPG (x, y, z), dimensions, radius, total points in sphere, time for kernel to run");
	print("%d,(%d, %d, %d),%ld,%.4f,%ld,%.4f,",tpb, bpg, bpg1, bpg2, nDimensions, radius, totalPoints, time);
}

long powerLong(long base, long exponent) {
	long result = 1;
	for (int i = 0; i < exponent; ++i) {
		result *= base;
	}
	return result;
}

// Kernel for calculating point in ndim sphere
__global__ void gpuCountPoints(ULL nPointsToTest, double radiusSquared, 
							   ULL halfBase, ULL base, 
							   ULL nDimensions, char* record,
							   ULL* count) {

	ULL blockId = blockIdx.y * gridDim.x + blockIdx.x;
	ULL id = blockId * blockDim.x + threadIdx.x;

	if (id < nPointsToTest) {
		ULL index[MAX_DIMENSIONS];
		for (int i = 0; i < MAX_DIMENSIONS; i++) {
			index[i] = halfBase;
		}

		ULL point = id;
		ULL distance = 0;
		for (int i = 0; i < nDimensions; i++) {
			index[i] = point % base;
			point = point / base;

			long long difference = index[i] - halfBase;
			ULL differenceSquared = (difference * difference);
			distance += differenceSquared;
		}

		if (distance < radiusSquared) {
			record[id] = 1;
		}

		atomicAdd(count, record[id]);
	}
}

// Runs a gpu kernel to work out how many integer points are in an ndimensional sphere. 
// Can manually set testCase to work through some of the test cases defined or 
// run the program with the test case as an argument. There are 27 test cases. 
int main(int argc, char** argv) {
	// cuda event creation for timing the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Determine some test cases 
	ULL dimensions[] = {1, 2, 3, 
						1, 1, 1, 
						2, 2, 2,
						3, 3, 3,
						4, 4, 4,
						5, 5, 5,
						6, 6, 6,
						7, 7, 7,
						8, 8, 8};
	double radii[] = {25.5, 2.05, 1.5, 
					  2.05, 5.05, 10.05,
					  2.05, 5.05, 10.05,
					  2.05, 5.05, 10.05,
					  2.05, 5.05, 10.05,
					  2.05, 5.05, 10.05,
					  2.05, 5.05, 10.05,
					  2.05, 5.05, 10.05,
					  2.05, 5.05, 10.05};

	// Use the 2 dimensional test case if none is selected
	int testCase = 1;

	if (argc == 2) {
		testCase = atoi(argv[1]);
		debug("TestCase is %d", testCase);
	}

	// initialise important variables
	ULL halfBase = static_cast<ULL>(floor(radii[testCase]));
	ULL base = 2 * halfBase + 1;
	ULL nPointsToTest = powerLong(base, dimensions[testCase]);
	double radiusSquared = radii[testCase] * radii[testCase];

	debug("gpu settings tc:%d: nPointsToTest: %ld, nDimensions: %ld, radius: %.2f, base: %ld", 
	testCase, nPointsToTest, dimensions[testCase], radii[testCase], base);

	// get the size to transfer to the device and initialise the array that will be sent
	ULL nBytesOutsideRecord = sizeof(char) * nPointsToTest;
	ULL nBytesCount = sizeof(ULL);
	debug("nBytesOutsideRecord: %ld, nBytesCount: %ld", nBytesOutsideRecord, nBytesCount);
	char* record = (char *)malloc(nBytesOutsideRecord);
	ULL count = 0;
	for (int i = 0; i < nPointsToTest; ++i) {
		record[i] = 0;
	}
	
	debug("Allocating memory done");

	// pointers for gpu arrays
	char* gpuRecord;
	ULL* gpuCount;

	// allocate memory on device (gpu)
	cudaMalloc(&gpuRecord, nBytesOutsideRecord);
	cudaMalloc(&gpuCount, nBytesCount);

	// copy over data to device
	cudaMemcpy(gpuRecord, record, nBytesOutsideRecord, cudaMemcpyHostToDevice);
	cudaMemcpy(gpuCount, &count, nBytesCount, cudaMemcpyHostToDevice);

	int nThreads = 1024;
	int nBlocksNeeded = (nPointsToTest + nThreads - 1) / nThreads;
	int nBlocks = nBlocksNeeded;
	int nBlocks1 = 1;
	int nBlocks2 = 1;
	if (nBlocks > 65535) {
		nBlocks = 65535;
		nBlocks1 = ceil(nBlocksNeeded/65535.0f);
	}

	debug("Block dimensions (%d, %d, %d)", nBlocks, nBlocks1, 1);

	dim3 blockDimensions(nThreads, 1, 1);
	dim3 gridDimensions(nBlocks, nBlocks1, nBlocks2);

	// run the kernel with timing
	cudaEventRecord(start);
	gpuCountPoints<<<gridDimensions, blockDimensions>>>(nPointsToTest, radiusSquared, 
														halfBase, base, 
														dimensions[testCase], gpuRecord,
														gpuCount);

	cudaMemcpy(record, gpuRecord, nBytesOutsideRecord, cudaMemcpyDeviceToHost);
	cudaMemcpy(&count, gpuCount, nBytesCount, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float time;
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // free up memory
	cudaFree(gpuRecord);
	cudaFree(gpuCount);
	free(record);

	// printTestResults(dimensions[testCase], radii[testCase], count);
	printCSVTestResults(nThreads, nBlocks, nBlocks1, nBlocks2, dimensions[testCase], radii[testCase], count, time);
	debug("Kernel took %fms", time);
	
	return 0;
}