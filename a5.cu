#include "utilities.h"
#include <vector>

using namespace std;

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

void runSequentialTestCases(long *dimensions, double* radii) {
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

__device__ long getDimensionalValue(long point, long base, long dimension) {
	long result = 0;
	for (int i = 0; i < dimension; i++) {
		result = point % base;
		point = point / base;
	}
	return result; 
} 

__device__ int outside(long value, long halfBase, double radiusSquared) {
	long difference = value - halfBase;
	
	if (difference * difference < radiusSquared) {
		return 0;
	} else {
		return 1;	
	}	
}

__global__ void gpuCountPoints(long nPointsToTest, double radiusSquared, long halfBase, long base, int* outsideRecord) {

	long id = blockIdx.x * blockDim.x + threadIdx.x;
	long dimension = blockDim.y;

	if (id < nPointsToTest) {
		long dimensionalValue = getDimensionalValue(id, base, dimension);
		int isOutside = outside(dimensionalValue, halfBase, radiusSquared);
		atomicAdd(&outsideRecord[id], isOutside);
	}
}

int main(int argc, char** argv) {
	int nTests = 3;
	long dimensions[] = {1, 2, 3};
	double radii[] = {25.5, 2.05, 1.5};

	runSequentialTestCases(dimensions, radii);

	int testCase = 1;

	if (argc == 2) {
		testCase = atoi(argv[1]);
		print("TestCase is %d", testCase);
	}

	const long halfBase = static_cast<long>(floor(radii[testCase]));
	const long base = 2 * halfBase + 1;
	const long nPointsToTest = powerLong(base, dimensions[testCase]);
	const double radiusSquared = radii[testCase] * radii[testCase];

	debug("gpu settings tc:%d: nPointsToTest: %ld, nDimensions: %ld, radius: %f, base: %ld", 
		testCase, nPointsToTest, dimensions[testCase], radii[testCase], base);

	int nBytes = sizeof(int) * nPointsToTest;

	int* outsideRecord = (int *)malloc(nBytes);
	for (int i = 0; i < nPointsToTest; ++i) {
		outsideRecord[i] = 0;
	}

	int* gpuOutsideRecord;
	cudaMalloc(&gpuOutsideRecord, nBytes);
	cudaMemcpy(gpuOutsideRecord, outsideRecord, nBytes, cudaMemcpyHostToDevice);

	int nThreads = 32 * dimensions[testCase];
	int nBlocks = (nPointsToTest + nThreads - 1) / nThreads;

	dim3 blockDimensions(nThreads, dimensions[testCase], 1);
	dim3 gridDimensions(nBlocks, 1, 1);

	gpuCountPoints<<<gridDimensions, blockDimensions>>>(nPointsToTest, radiusSquared, halfBase, base, gpuOutsideRecord);

	cudaMemcpy(outsideRecord, gpuOutsideRecord, nBytes, cudaMemcpyDeviceToHost);

	cudaFree(gpuOutsideRecord);

	long count = 0;
	for (int i = 0; i < nPointsToTest; ++i) {
		if (outsideRecord[i] == 0) {
			count++;
		}
	}

	for (int i = 0; i < nPointsToTest; i++) {
		print("%d :: %d", i, outsideRecord[i]);
	}

	free(outsideRecord);

	print("Parallel test %d:", testCase);
	printTestResults(dimensions[testCase], radii[testCase], count);
	
	return 0;
}