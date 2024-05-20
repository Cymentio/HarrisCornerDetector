#ifndef HERITANCE_H
#define HERITANCE_H

#include "test.h"
#include <iostream>

class Sequential : public Test {
public:
    __device__ void test(void){
    printf("aha\n");
    // jadro<<<1,1>>>();
};
    __device__ void aaa();
};

// __device__ void Sequential::test(void){
//     printf("aha\n");
//     // jadro<<<1,1>>>();
// };

#endif