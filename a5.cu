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

long powerLong(long base, long exponent) {
	long result = 1;
	for (int i = 0; i < exponent; ++i) {
		result *= base;
	}
	return result;
}

void convert(long point, long base, vector<long>& index) {
	const long nDimensions = index.size();
	for (long i = 0; i < nDimensions; ++i) {
		index[i] = 0;
	}

	long id = 0;
	while (point != 0) {
		long remainder = point % base;
		point = point / base;
		index[id] = remainder;
		id++;
	}
}

long countPoints(long nDimensions, double radius) {
	const long halfBase = static_cast<long>(floor(radius));
	const long base = 2 * halfBase + 1;
	const double radiusSquared = radius * radius;
	const long nPointsToTest = powerLong(base, nDimensions);

	debug("countPoints():\nhalfBase: %ld\n, base: %ld\n, radiusSquared: %.2f\n, nPointsToTest: %ld", halfBase, base, radiusSquared, nPointsToTest);

	long count = 0;
	vector<long> index(nDimensions, 0);

	for (long point = 0; point < nPointsToTest; ++point) {
	    convert(point, base, index);
	    double testRadiusSquared = 0;
	    for (long dimension = 0; dimension < nDimensions; ++dimension) {
	        double difference = index[dimension] - halfBase;
	        testRadiusSquared += difference * difference;
	    }
	    if (testRadiusSquared < radiusSquared) {
	    	++count;
	    }
	}

	return count;
}

void runSequentialTestCases(ULL *dimensions, double* radii) {
	print("Sequential Tests ---------------");
	for (int i = 0; i < 3; i++) {
		long totalPoints = countPoints(dimensions[i], radii[i]);
		printTestResults(dimensions[i], radii[i], totalPoints);
	}
	print("Sequential Tests Over ----------\n");
}

__device__ void convert(long point, long base, long* index, long nDimensions) {
	// Ensure array initialised
	for (int i = 0; i < nDimensions; ++i) {
		index[i] = 0;
	}

	long i = 0;
	while (point != 0) {
		long remainder = point % base;
		point = point / base;
		index[i] = remainder;
		i++;
	}
}

__device__ ULL getDimensionalValue(ULL point, ULL base, ULL dimension) {
	ULL result = 0;
	for (int i = 0; i < dimension; i++) {
		result = point % base;
		point = point / base;
	}
	return result; 
} 

__device__ void determineOutside(ULL id, ULL dimension, 
								 ULL* pointsLength, double radiusSquared, 
								 int* record) {
	if (dimension == 1) {
		if (pointsLength[id] < radiusSquared) {
			record[id] = 0;
		} else {
			record[id] = 1;
		}
	}
}

// __device__ void addComponentToLength(ULL id, ULL value, ULL halfBase, ULL* pointsLength) {
// 	long long difference = value - halfBase;
// 	// ULL differenceSquared = (difference * difference);
// 	// atomicAdd(&pointsLength[id], differenceSquared);
// 	ULL differenceSquared = (difference * difference);
// 	atomicAdd(&pointsLength[id], difference);
// }

__global__ void gpuCountPoints(ULL nPointsToTest, double radiusSquared, 
							   ULL halfBase, ULL base, 
							   ULL nDimensions, int* record) {

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
			// record[i] = differenceSquared;
		}

		if (distance < radiusSquared) {
			record[id] = 1;
		}



		// record[id] = id;
	}
}

int main(int argc, char** argv) {

	// Test variables that I have the answers to
	int nTests = 3;
	ULL dimensions[] = {1, 2, 3};
	double radii[] = {25.5, 2.05, 1.5};

	// Sequential answers
	runSequentialTestCases(dimensions, radii);

	// Use the 2 dimensional test case if none is selected
	int testCase = 1;

	if (argc == 2) {
		testCase = atoi(argv[1]);
		print("TestCase is %d", testCase);
	}

	// initialise important variables
	const ULL halfBase = static_cast<ULL>(floor(radii[testCase]));
	const ULL base = 2 * halfBase + 1;
	const ULL nPointsToTest = powerLong(base, dimensions[testCase]);
	const double radiusSquared = radii[testCase] * radii[testCase];

	debug("gpu settings tc:%d: nPointsToTest: %ld, nDimensions: %ld, radius: %.2f, base: %ld", 
		testCase, nPointsToTest, dimensions[testCase], radii[testCase], base);

	// get the size to transfer to the device and initialise the array that will be sent
	int nBytesOutsideRecord = sizeof(int) * nPointsToTest;
	int* record = (int *)malloc(nBytesOutsideRecord);
	for (int i = 0; i < nPointsToTest; ++i) {
		record[i] = 0;
	}

	int* gpuRecord;

	cudaMalloc(&gpuRecord, nBytesOutsideRecord);

	cudaMemcpy(gpuRecord, record, nBytesOutsideRecord, cudaMemcpyHostToDevice);

	int nThreads = 256;
	int nBlocks = (nPointsToTest + nThreads - 1) / nThreads;

	dim3 blockDimensions(nThreads, 1, 1);
	dim3 gridDimensions(nBlocks, 1, 1);

	gpuCountPoints<<<gridDimensions, blockDimensions>>>(nPointsToTest, radiusSquared, halfBase, base, dimensions[testCase], gpuRecord);

	cudaMemcpy(record, gpuRecord, nBytesOutsideRecord, cudaMemcpyDeviceToHost);

	cudaFree(gpuRecord);

	long count = 0;
	for (int i = 0; i < nPointsToTest; ++i) {
		if (record[i] == 1) {
			count++;
		}
	}

	print("Ok, now for outside record, remember radius^2 is %f", radiusSquared);
	printArray(record, nPointsToTest);

	free(record);

	print("Parallel test %d:", testCase);
	printTestResults(dimensions[testCase], radii[testCase], count);
	
	return 0;
}