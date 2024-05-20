#include <iostream>
#include <cstdlib>
#include "test.h"
// #include "heritance.h"

int main(int argc, char* argv[]) {
    if (argc == 2) {
        printf("%s",argv[1]); // 1 - nazwa wejściowego obrazu
    } else {
        printf("podaj poprawna liczbę argumentow: 1\n");
        exit(1);
    }
    HarrisCornerDetector* t = new SeqHarrisCornerDetector();
    printf("eee\n");
    t->setImgPath(argv[1]);
    printf("aaa\n");
    detectCorners(t);
    printf("\n");
    t = new ParHarrisCornerDetector(1, 1);
    detectCorners(t);
//  std::clog << "Wersja: "  << CUDA_VERSION << std::endl;
//  std::clog << "Wersja: "  << CUDART_VERSION << std::endl;;
    return 0;
}