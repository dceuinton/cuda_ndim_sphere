
#ifndef UTILITIES
#define UTILITIES

// #define DEBUG

#include <cstdarg>
#include <stdio.h>
#include <iostream>

void debug(const char* format, ...) {
    #ifdef DEBUG
    va_list args;
    va_start(args, format);
    vfprintf(stdout, "DEBUG :: ", NULL);
    vfprintf(stdout, format, args);
    vfprintf(stdout, "\n", NULL);
    va_end(args);
    fflush(stdout);
    #endif
}

// basic print function
void print(const char* format, ...) {
    va_list args;
    va_start(args, format);
    vfprintf(stdout, format, args);
    vfprintf(stdout, "\n", NULL);
    va_end(args);
    fflush(stdout);
}

#endif 