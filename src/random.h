#ifndef RANDOM_H
#define RANDOM_H
#include <curand.h>

unsigned int* curand_int( curandGenerator_t& gen, size_t n );

float* curand_uniform( curandGenerator_t& gen, size_t n );

float* curand_normal( curandGenerator_t& gen, size_t n, float mean, float stddev );

#endif
