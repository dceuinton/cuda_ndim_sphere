#include "utilities.h"

#include <iostream>

int main(int argc, char** argv) {
	print("Hello world!");

	unsigned long long v1 = 5;
	unsigned long long v2 = 7;
	long long thing = v1 - v2;
	unsigned long long thing1 = thing * thing;

	std::cout << thing << std::endl;
	std::cout << thing1 << std::endl;

	debug("Debug statement");
}