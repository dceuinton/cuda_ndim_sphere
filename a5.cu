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

__global__ void gpuCountPoints(long nPointsToTest, long nDimensions, double radius, long base) {

}

int main(int argc, char** argv) {
	int nTests = 3;
	long dimensions[] = {1, 2, 3};
	double radii[] = {25.5, 2.05, 1.5};

	runSequentialTestCases(dimensions, radii);

	int testCase = 0;

	const long halfBase = static_cast<long>(floor(radii[testCase]));
	const long base = 2 * halfBase + 1;
	const long nPointsToTest = powerLong(base, dimensions[testCase]);

	debug("gpu settings tc:%d: nPointsToTest: %ld, nDimensions: %ld, radius: %f, base: %ld", 
		testCase, nPointsToTest, dimensions[testCase], radii[testCase], base);

	
	
	return 0;
}