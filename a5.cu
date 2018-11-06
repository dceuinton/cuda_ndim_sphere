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

void printCSVTestResults(int tpb, int bpg, long nDimensions, double radius, long totalPoints, float time) {
	print("\nCSV Results:");
	print("TPB, BPG, dimensions, radius, total points in sphere, time for kernel to run");
	print("%d,%d,%ld,%.4f,%ld,%.4f,",tpb, bpg, nDimensions, radius, totalPoints, time);
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
							   ULL nDimensions, int* record,
							   int* count) {

	ULL id = blockIdx.x * blockDim.x + threadIdx.x;

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
		print("TestCase is %d", testCase);
	}

	// initialise important variables
	ULL halfBase = static_cast<ULL>(floor(radii[testCase]));
	ULL base = 2 * halfBase + 1;
	ULL nPointsToTest = powerLong(base, dimensions[testCase]);
	double radiusSquared = radii[testCase] * radii[testCase];

	debug("gpu settings tc:%d: nPointsToTest: %ld, nDimensions: %ld, radius: %.2f, base: %ld", 
	testCase, nPointsToTest, dimensions[testCase], radii[testCase], base);

	// get the size to transfer to the device and initialise the array that will be sent
	int nBytesOutsideRecord = sizeof(int) * nPointsToTest;
	int nBytesCount = sizeof(int);
	int* record = (int *)malloc(nBytesOutsideRecord);
	int count = 0;
	for (int i = 0; i < nPointsToTest; ++i) {
		record[i] = 0;
	}

	int* gpuRecord;
	int* gpuCount;

	cudaMalloc(&gpuRecord, nBytesOutsideRecord);
	cudaMalloc(&gpuCount, nBytesCount);

	cudaMemcpy(gpuRecord, record, nBytesOutsideRecord, cudaMemcpyHostToDevice);
	cudaMemcpy(gpuCount, &count, nBytesCount, cudaMemcpyHostToDevice);

	int nThreads = 256;
	int nBlocks = (nPointsToTest + nThreads - 1) / nThreads;

	dim3 blockDimensions(nThreads, 1, 1);
	dim3 gridDimensions(nBlocks, 1, 1);

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

	cudaFree(gpuRecord);
	cudaFree(gpuCount);

	free(record);

	// printTestResults(dimensions[testCase], radii[testCase], count);
	printCSVTestResults(nThreads, nBlocks, dimensions[testCase], radii[testCase], count, time);
	print("Kernel took %fms", time);
	
	return 0;
}