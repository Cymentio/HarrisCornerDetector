#include "test.h"
// #include "heritance.h"
#include <iostream>
#include <typeinfo>
#include <cuda.h>
#include <windows.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"

void detectCorners(HarrisCornerDetector* t){
    printf("aha\n");
    // s.aaa();
    // delete s;
    // printf(typeid(*t).name())
    t->start();
    printf("ehe\n");
    delete t;
}

void HarrisCornerDetector::setImgPath(const char* path){
    img_path = path;
}

HarrisCornerDetector::~HarrisCornerDetector(){
}

SeqHarrisCornerDetector::SeqHarrisCornerDetector(){
    img_path = "";
}

void SeqHarrisCornerDetector::start(){
    printf("start sequential\n");
    printf("%s", img_path);
    unsigned char* img = stbi_load(img_path, &width, &height, &channels, 3);
    if(img == NULL) {
         printf("Error in loading the image\n");
         exit(1);
    }
    printf("Loaded image with a width of %dpx, a height of %dpx and %d channels\n", width, height, channels);
    if(channels != 3 && channels != 1) {
         printf("img not rgb\n");
         exit(1);
    }
    unsigned char* img_grayscale= imgToGrayscale(img);

    stbi_write_png("1.png", width, height, 1, img_grayscale, width);
    delete[] img_grayscale;
    
    stbi_image_free(img);
    // jadro<<<1,1>>>();
}

ParHarrisCornerDetector::ParHarrisCornerDetector(unsigned int n, unsigned int d){
    img_path = "";
    blocks_n = n;
    block_dim = d;
    width = 0;
    height = 0;
    channels = 0;
}

void ParHarrisCornerDetector::start(){
    printf("start parallel\n");
    // aaa<<<blocks_n, block_dim>>>(this);
    // Sleep(2000);
    // aaa<<<blocks_n, block_dim>>>(this);
    // printf("eee\n");
    // jadro<<<1,1>>>();
}

unsigned char* SeqHarrisCornerDetector::imgToGrayscale(unsigned char* img){
    unsigned char* img_grayscale = new unsigned char[width*height];
    if (channels == 1) {
        for(int i=0; i< width*height; i++)
            img_grayscale[i] = img[i];
        return img_grayscale;
    }
    for (int i=0; i<height*width;i++){
        img_grayscale[i] = img[i * channels] * 0.299 + img[i * channels + 1] * 0.587 + img[i * channels+2] * 0.114;
    }
    return img_grayscale;
}

unsigned char* ParHarrisCornerDetector::imgToGrayscale(unsigned char* img){
    return nullptr;
}

__device__ void ParHarrisCornerDetector::dev(){
    printf("device\n");
}

__global__ void aaa(ParHarrisCornerDetector* h){
    printf("aaaaaaaaaaaaaaaaaaaaaa\n");
    h->dev();
}