#include <iostream>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <typeinfo>
#include <windows.h>
#include <chrono>
#include <omp.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"

#define THREADS_PER_BLOCK 128

__constant__ int cmask[25];
__constant__ int cgx[9];
__constant__ int cgy[9];
__constant__ int cm[4];

int* img_to_grayscale(unsigned char* img, int width, int height, int channels){
    int* img_grayscale = new int[width*height];
    int i;

    if (channels == 1) {
        // # pragma omp parallel for private(i)
        for(i=0; i< width*height; i++)
            img_grayscale[i] = img[i];
        return img_grayscale;
    }
    // # pragma omp parallel for private(i)
    for (i=0; i<height*width;i++){
        img_grayscale[i] = img[i * channels] * 0.299 + img[i * channels + 1] * 0.587 + img[i * channels+2] * 0.114;
    }

    return img_grayscale;
}

__global__ void grayscale_parallel(int n, int channels, int * img_grayscale, unsigned char * img){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        if(channels == 1) img_grayscale[i] = img[i];
        else {
            img_grayscale[i] = img[i * channels] * 0.299 + img[i * channels + 1] * 0.587 + img[i * channels+2] * 0.114;
        }
        // i += blockDim.x * gridDim.x;
    }
    return;
}

long int gaussian_value(long int* img, int width, int y, int x, int mask[], int mask_size, int half_mask){
    long long int result = 0;
    int x1, y1;
    for(y1 = 0; y1 < mask_size; y1++){
        for(x1 = 0; x1 < mask_size; x1++){
            result += mask[y1 * mask_size + x1] * img[(y + y1) * (width + mask_size - 1) + (x + x1)];
        }
    }
    return (long int) result / 273;
}

__device__ long int dev_gaussian_value(long int* img, int width, int y, int x, int mask[], int mask_size, int half_mask, int n_pad, int idy){
    long long int result = 0;
    int x1, y1;
    for(y1 = 0; y1 < mask_size; y1++){
        for(x1 = 0; x1 < mask_size; x1++){
            result += mask[y1 * mask_size + x1] * img[(y + y1) * (width + mask_size - 1) + (x + x1) + idy * n_pad];
        }
    }
    return (long int) result / 273;
}

// reflection padding because zero padding (constant value padding) can add corners to image borders
int* reflection_padding(int* img, int width, int height, int padding_size){ // padding_size - increase of width and height in pixels (both sides): (mask_dim - 1) / 2
    int* img_padded = new int[(width + 2*padding_size)*(height + 2*padding_size)]; // img will be reflection padded

    for (int y=0; y < height + 2*padding_size; y++){
        int y_diff = 0;
        if (y < padding_size){
            y_diff = (padding_size - y) * 2; // [hp = padding_size/2] 0 (-hp original) -> hp + 1 (hp - 1 in original); hp - 1 (-1 original) -> hp (0 in original)
        }
        if (y >= height + padding_size){
            y_diff = (height + padding_size - 1 - y) * 2;
        }
        for (int x=0; x < width + 2*padding_size; x++){
            int x_diff = 0;
            if (x < padding_size){ // skipped variable for x_diff
                x_diff = (padding_size - x) * 2;
            }
            if (x >= width + padding_size){
                x_diff = (width + padding_size - 1 - x) * 2;
            }
            img_padded[y * (width + 2 * padding_size) + x] = img[(y - padding_size + y_diff) * width + x - padding_size + x_diff]; // n + 2 -> n in original for y and x
        }
    }

    return img_padded;
}

// long int version
long int* reflection_padding(long int* img, int width, int height, int padding_size){ // adding_size - increase of width and height in pixels (both sides) (mask_dim - 1) / 2
    long int* img_padded = new long int[(width + 2*padding_size)*(height + 2*padding_size)]; // img will be reflection padded
    int y, x;

    for (y=0; y < height + 2*padding_size; y++){
        int y_diff = 0;
        if (y < padding_size){
            y_diff = (padding_size - y) * 2; // [hp = padding_size/2] 0 (-hp original) -> hp + 1 (hp - 1 in original); hp - 1 (-1 original) -> hp (0 in original)
        }
        if (y >= height + padding_size){
            y_diff = (height + padding_size - 1 - y) * 2;
        }
        for (x=0; x < width + 2*padding_size; x++){
            int x_diff = 0;
            if (x < padding_size){ // skipped variable for x_diff
                x_diff = (padding_size - x) * 2;
            }
            if (x >= width + padding_size){
                x_diff = (width + padding_size - 1 - x) * 2;
            }
            img_padded[y * (width + 2 * padding_size) + x] = img[(y - padding_size + y_diff) * width + x - padding_size + x_diff]; // n + 2 -> n in original for y and x
        }
    }

    return img_padded;
}


int operator_value(int* img, int width, int y, int x, int mask[], int mask_size, int half_mask){
    int result = 0;
    int y1, x1;

    for(y1 = 0; y1 < mask_size; y1++){
        for(x1 = 0; x1 < mask_size; x1++){
            result += mask[y1 * mask_size + x1] * img[(y + y1) * (width + mask_size - 1) + (x + x1)];
        }
    }

    return result;
}

__device__ int dev_operator_value(int* img, int width, int y, int x, int mask[9], int mask_size){
    int result = 0;
    int y1, x1;

    for(y1 = 0; y1 < mask_size; y1++){
        for(x1 = 0; x1 < mask_size; x1++){
            result += mask[y1 * mask_size + x1] * img[(y + y1) * (width + mask_size - 1) + (x + x1)];
        }
    }

    return result;
}

__global__ void global_operator_value(int* img, int width, int y, int x, int mask[], const int mask_size, int n, int* result){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int y1 = i / mask_size;
    int x1 = i % mask_size;
    if (i < n) {
        result[i] = mask[y1 * mask_size + x1] * img[(y + y1) * (width + mask_size - 1) + (x + x1)];
    }

    return;
}

__global__ void reduce_val(int* result, int n, int* reduce_array){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    for(int s = 1; s < n; s*=2){
        if (i < n && i % (2*s) == 0) {
        reduce_array[i] += result[i + s];
        }
        __syncthreads();
    }

    return;
}

void operator_values(int* img, int width, int y, int x, int mask_x[], int mask_y[], int mask_size, int half_mask, int* ix, int* iy){
    int result_x = 0, result_y = 0;
    int y1, x1;

    for(y1 = 0; y1 < mask_size; y1++){
        for(x1 = 0; x1 < mask_size; x1++){
            result_x += mask_x[y1 * mask_size + x1] * img[(y + y1) * (width + mask_size - 1) + (x + x1)];
            result_y += mask_y[y1 * mask_size + x1] * img[(y + y1) * (width + mask_size - 1) + (x + x1)];
        }
    }

    ix[y * width + x] = result_x;
    iy[y * width + x] = result_y;
    return;
}

// does convolution on image using sobel operators and returns products of derivatives (sobel operator results)
long int** compute_derivatives(int* img, int width, int height){ 
    int* ix = new int[width * height];
    int* iy = new int[width * height];
    int x, y, i;
    // Sobel operators
    int gx[9] = {
        1, 0, -1,
        2, 0, -2,
        1, 0, -1
    };
    int gy[9] = {
        1, 2, 1,
        0, 0, 0,
        -1, -2, -1
    };

    // printf("padding\n");
    int* img_padded = reflection_padding(img, width, height, 1);
    // printf("padded\n");

    long int** i_arr = new long int*[3]; // ixix iyiy ixiy

    // #pragma omp parallel private(x, y, i) shared(height, width, gx, gy, i_arr, ix, iy)
    // {
        // #pragma omp for
        for (y = 0; y < height; y++){
            for (x=0; x < width; x++){
                // operator_values(img_padded, width, y, x, gx, gy, 3, 1, ix, iy); // ix iy
                ix[y * width + x] = operator_value(img_padded, width, y, x, gx, 3, 1); // ix
                iy[y * width + x] = operator_value(img_padded, width, y, x, gy, 3, 1); // iy
            }
        }

        // #pragma omp single
        {
        for (i = 0; i < 3; i++){
                    i_arr[i] = new long int[width * height];
            }
        }
        // #pragma omp barrier

        // #pragma omp for
        for (y = 0; y < height; y++){
            for (x=0; x < width; x++){ // after loop above
                i_arr[0][y * width + x] = ix[y * width + x] * ix[y * width + x]; // ixix
                i_arr[1][y * width + x] = iy[y * width + x] * iy[y * width + x]; // iyiy
                i_arr[2][y * width + x] = ix[y * width + x] * iy[y * width + x]; // ixiy
            }
        }
    // }
    delete[] ix;
    delete[] iy;

    delete[] img_padded;
    return i_arr;
}

// // does convolution on image using sobel operators and returns products of derivatives (sobel operator results)
__global__ void derivatives_parallel(int* img, int width, int height, int n, long int* i_arr, const int gx[9], const int gy[9], int * img_padded){ 
    int ix, iy, x, y;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        x = i % width;
        y = i / width;
        ix = dev_operator_value(img_padded, width, y, x, cgx, 3); // ix
        iy = dev_operator_value(img_padded, width, y, x, cgy, 3); // iy
        i_arr[y * width + x] = ix * ix; // ixix
        i_arr[y * width + x + 1 * n] = iy * iy; // iyiy
        i_arr[y * width + x + 2 * n] = ix * iy; // ixiy
        // i += blockDim.x * gridDim.x;
    }

    return;
}

long int* gaussian_filter(long int* img, int width, int height){ // does convolution on image using mask
    int x, y;

    int mask[25] = {
        1,  4,  7,  4, 1,
        4, 16, 26, 16, 4,
        7, 26, 41, 26, 7,
        4, 16, 26, 16, 4,
        1,  4,  7,  4, 1
    };
    long int* img_padded = reflection_padding(img, width, height, 2);
    long int* img_gaussian = new long int[width*height];
    // # pragma omp parallel for private(y, x)
    for (y = 0; y < height; y++){
        for (x = 0; x < width; x++){
            img_gaussian[y * width + x] = gaussian_value(img_padded, width, y, x, mask, 5, 2);
        }
    }

    delete[] img_padded;
    return img_gaussian;
}

__global__ void gaussian_parallel(long int* img, int width, int height, int n, int n_pad, int mask[25], long int *img_gaussian){ // does convolution on image using mask
    int x, y;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        x = i % width;
        y = i / width;
        img_gaussian[i + n * blockIdx.y] = dev_gaussian_value(img, width, y, x, cmask, 5, 2, n_pad, blockIdx.y);
        // i += blockDim.x * gridDim.x;
    }
    // # pragma omp parallel for private(y, x)
    // for (y = 0; y < height; y++){
    //     for (x = 0; x < width; x++){
    //         // img_gaussian[y * width + x] = gaussian_value(img_padded, width, y, x, mask, 5, 2);
    //     }
    // }

    return;
}

long long int* pixel_response(long int** i_arr, int width, int height, float k){ // 1 if det(M) - k*tr(M)^2 above threshold 0 if below for each pixel
    // k - constant <0.04-0.06>
    // n_dim * n_dim window (neighborhood), only if center pixel is has the highest value in neghborhood it can give 1 as response
    int m[4] = {
        0, 2, // ixix ixiy
        2, 1  // ixiy iyiy
    };
    long long int* r_arr = new long long int[width*height];
    int x, y;
    long long det, tr;

    // #pragma omp parallel for shared(height, width, i_arr, r_arr) private(y, x, det, tr)
    for (y = 0; y < height; y++){
        for (x=0; x < width; x++){
            det = i_arr[m[0]][y * width + x] * i_arr[m[3]][y * width + x] - i_arr[m[1]][y * width + x] * i_arr[m[2]][y * width + x];
            tr = i_arr[m[0]][y * width + x] + i_arr[m[3]][y * width + x];
            r_arr[y * width + x] = (long long) (det - k * tr * tr);             
        }
    }

    return r_arr;
}

__global__ void pixel_response_parallel(long int* i_arr, int width, int height, int n, float k, long long int* r_arr){ // 1 if det(M) - k*tr(M)^2 above threshold 0 if below for each pixel
    // k - constant <0.04-0.06>
    // n_dim * n_dim window (neighborhood), only if center pixel is has the highest value in neghborhood it can give 1 as response
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        long long det, tr;
        det = i_arr[i] * i_arr[n + i] - i_arr[2 * n + i] * i_arr[2 * n + i];
        tr = i_arr[i] + i_arr[n + i];
        r_arr[i] = (long long) (det - k * tr * tr);             
    }

    return;
}

struct corner {
    int index;
    long long value;
};

bool compare_corner(const corner &a, const corner &b)
{
    return a.value > b.value;
}

bool* threshold_response(long long* r_arr, int width, int height, long long threshold, int n_dim, int max_corners){ // 1 if value above threshold 0 if below for each pixel
    // (2*n_dim + 1) * (2*n_dim + 1) window (neighborhood), only if center pixel is has the highest value in neghborhood it can give 1 as response
    // n_dim - how many pixel from center to each side is neighborhood
    int arr_size = width * height;
    bool* t_arr = new bool[arr_size];
    for (int i = 0; i < arr_size; i++){
        t_arr[i] = false;
    }
    int x, y, y1, x1;
    long long r;

    // kombinacja liniowa pikselów o response > threshold dla każdego sąsiedztwa n_dim*n_dim prostokątów równej wielkości z obrazka to corner
    int x_step = int(width / n_dim);
    int y_step = int(height / n_dim);

    std::vector<corner> corners;

    for (y = 0; y <= height - y_step; y+= y_step){
        for (x=0; x <= width - x_step; x+= x_step){
            long long x_mean = 0;
            long long y_mean = 0;
            long long sum_weights = 0;
            long long sum_response = 0;
            for (y1 = y; y1 < y + y_step; y1++){
                // #pragma omp parallel for default(shared) reduction(+:sum_weights, x_mean, y_mean, sum_response) private(x1, r)
                for (x1 = x; x1 < x + x_step; x1++){
                    r = r_arr[y1*width + x1];
                    if (threshold < r){
                        sum_response += r;
                        sum_weights += (r / threshold);
                        x_mean += (r / threshold) * x1;
                        y_mean += (r / threshold) * y1;
                    }
                }
            } 
            if (sum_weights != 0){               
                int corner_x = x_mean / sum_weights;
                int corner_y = y_mean / sum_weights;
                corner c;
                c.index = corner_y*width + corner_x;
                c.value = sum_response;
                corners.push_back(c);
            }           
        }
    }

    // leave max_corners corners with highest sum of values
    std::sort(corners.begin(), corners.end(), compare_corner);
    if(corners.size()>=max_corners) 
    corners.resize(max_corners);
    for (auto c: corners){
        t_arr[c.index] = true;
    }

    return t_arr;
}

void color_corners(unsigned char* img, int width, int height, bool* t_arr, int channels, int corner_size){ 
    // adds red crosses to corners
    int x, y;

    // #pragma omp parallel for
    for (y = 0; y < height; y++){
        for (x=0; x < width; x++){
            if (t_arr[y * width + x] == true){ // if corner add cross
                if (channels == 1) { // if grayscale black corners
                    for(int i= -corner_size; i <= corner_size; i++){
                        int index1 = (y + i) * width + x;
                        int index2 = y  * width + (x + i);
                        if (index1 >= 0 && index1 < width * height) // pixel in img borders
                            img[index1] = 255;
                        if (index2 >= 0 && index2 < width * height) // pixel in img borders
                            img[index2] = 255;
                    }
                    continue;                    
                }
                for(int i= -corner_size; i <= corner_size; i++){ // 3 channels
                    int index1 = (y + i) * width + x;
                    int index2 = y  * width + (x + i);
                    if (index1 >= 0 && index1 < width * height){ // pixel in img borders
                        img[index1 * channels] = 255; // red
                        img[index1 * channels + 1] = 0; // green
                        img[index1 * channels + 2] = 0; // blue
                    }
                    if (index2 >= 0 && index2 < width * height){ // pixel in img borders
                        img[index2 * channels] = 255; // red
                        img[index2 * channels + 1] = 0; // green
                        img[index2 * channels + 2] = 0; // blue
                    }
                }
            }
        }
    }
}

void detect_corners_seq(const char* img_path, long long threshold, int n_dim, float k, int max_corners, int cross_size){
    std::chrono::time_point <std::chrono::system_clock> start, end;
    int width, height, channels;
    int i;
    int* img_grayscale;
    long int** i_arr;
    printf("start sequential\n");
    printf("%s\n", img_path);
    unsigned char* img = stbi_load(img_path, &width, &height, &channels, 3);

    if(img == NULL) {
         printf("Error in loading the image\n");
         exit(1);
    }
    printf("Loaded image with a width of %dpx, a height of %dpx and %d channels\n", width, height, channels);
    if(channels != 3 && channels != 1 && channels != 4) {
         printf("img not rgb or grayscale\n");
         exit(1);
    }
    if (channels == 4)
        channels = 3;
    start = std::chrono::system_clock::now();


    // printf("grayscale\n");
    img_grayscale = img_to_grayscale(img, width, height, channels); // compute grayscale from img

    // printf("derivatives\n");
    i_arr = compute_derivatives(img_grayscale, width, height); // ixix iyiy ixiy - products of derivatives ix, iy(results of sobel operators gx, gy)

    // printf("gaussian\n");
    for (i = 0; i < 3; i++){
        long int* img_gaussian = gaussian_filter(i_arr[i], width, height); // gaussian filter for ixix iyiy ixiy
        delete[] i_arr[i];
        i_arr[i] = img_gaussian;
    }
    // printf("before response\n");

    long long int* r_arr = pixel_response(i_arr, width, height, k); // response function (k constant <0.04-0.06>)
    
    // printf("after response\n");

    bool* t_arr = threshold_response(r_arr, width, height, threshold, n_dim, max_corners); // which points on img are corners

    // for (int y = 0; y < height; y++){ // print corners coords
    //     for (int x=0; x < width; x++){
    //         if (t_arr[y * width + x] == true)
    //             printf("(%d, %d)\n", x, y);
    //     }
    // }

    // printf("coloring\n");
    color_corners(img, width, height, t_arr, channels, cross_size);
    for (i = 0; i < 3; i++){
        delete[] i_arr[i];
    }
    delete[] i_arr;
    delete[] r_arr;
    delete[] t_arr;

    end = std::chrono::system_clock::now();
    std::chrono::duration<double> computation_time = end - start;
    printf("computation time: %lf seconds\n", computation_time.count());
    printf("finished\n");

    // stbi_write_png("1.png", width, height, 1, img_grayscale, width);
    stbi_write_png("2.png", width, height, channels, img, width * channels);
    delete[] img_grayscale;
    
    stbi_image_free(img);
    return;
}

__global__ void test(int n){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    while (index < n) {
        if(index < n) printf("i: %d\n", index);
        index += blockDim.x * gridDim.x;
    }
    return;
}

__host__ void detect_corners_par(const char* img_path, long long threshold, int n_dim, float k, int max_corners, int cross_size){
    std::chrono::time_point <std::chrono::system_clock> start, end;
    int width, height, channels;
    int * img_grayscale, * dev_grayscale;
    long int** i_arr;
    printf("start cuda\n");
    printf("%s\n", img_path);
    unsigned char* img = stbi_load(img_path, &width, &height, &channels, 3);
    unsigned char* dev_img;

    if(img == NULL) {
         printf("Error in loading the image\n");
         exit(1);
    }
    printf("Loaded image with a width of %dpx, a height of %dpx and %d channels\n", width, height, channels);
    if(channels != 3 && channels != 1 && channels != 4) {
         printf("img not rgb or grayscale\n");
         exit(1);
    }
    if (channels == 4)
        channels = 3;
    start = std::chrono::system_clock::now();
    // printf("grayscale\n");
    int num_blocks = ceil( 1.0*(width * height) / THREADS_PER_BLOCK );
    printf("num_blocks: %d", num_blocks);
    long long sum = 0;
    img_grayscale = new int[width * height];
    int n = width * height;
    cudaMalloc((void**) &dev_grayscale, n * sizeof(int));
    cudaMalloc((void**) &dev_img, channels * n * sizeof(unsigned char));
    cudaMemcpy(dev_img, img, channels * n * sizeof(unsigned char), cudaMemcpyHostToDevice);
    grayscale_parallel<<<num_blocks, THREADS_PER_BLOCK>>>(n, channels, dev_grayscale, dev_img);
    cudaMemcpy(img_grayscale, dev_grayscale, n * sizeof(int), cudaMemcpyDeviceToHost);
    // cudaFree(dev_grayscale);
    cudaFree(dev_img);
    


    int * img_grayscale2 = img_to_grayscale(img, width, height, channels); // compute grayscale from img
// for (int i =n-1;i>0;i--){
//     if (img_grayscale2[i] != img_grayscale[i])
//     printf("%d: %d vs %d\n",i, img_grayscale[i], img_grayscale2[i]);
//     sum+=img_grayscale[i];
// }
    // printf(" %lld\n", sum);
    // Sobel operators
    const int gx[9] = {
        1, 0, -1,
        2, 0, -2,
        1, 0, -1
    };
    const int gy[9] = {
        1, 2, 1,
        0, 0, 0,
        -1, -2, -1
    };

    cudaMemcpyToSymbol(cgx, gx, 9 * sizeof(int));
    cudaMemcpyToSymbol(cgy, gy, 9 * sizeof(int));
    long int * i_arr2 = new long int [3 * n], * dev_arr;
    cudaMalloc((void**) &dev_arr, 3 * n * sizeof(long int));

    // printf("padding\n");
    int pad_size = 1;
    int* img_padded = reflection_padding(img_grayscale, width, height, pad_size);  // nieopłacalne na cuda
    int* dev_padded;
    // // printf("padded\n");
    
    // printf("derivatives\n");
    int n_pad = (width + 2*pad_size)*(height + 2*pad_size);
    // for (int i = 0; i< n_pad; i++)  printf("%ld ", img_padded[i]);
    cudaMalloc((void**) &dev_padded, n_pad * sizeof(int));
    cudaMemcpy(dev_padded, img_padded,  n_pad * sizeof(int), cudaMemcpyHostToDevice);
    derivatives_parallel<<<num_blocks, THREADS_PER_BLOCK>>>(dev_grayscale, width, height, n, dev_arr, gx, gy, dev_padded);
    cudaMemcpy(i_arr2, dev_arr, 3 * n * sizeof(long int), cudaMemcpyDeviceToHost);
    cudaFree(dev_padded);
    cudaFree(dev_grayscale);

    cudaFree(dev_arr);
    i_arr = new long int*[3];
    for (int i = 0; i< 3; i++){
        i_arr[i] = new long int[n]; 
        for (int j = 0; j< n; j++){
            i_arr[i][j] = i_arr2[i * n + j];
        }
    }


    long int **i_arr3 = compute_derivatives(img_grayscale2, width, height); // ixix iyiy ixiy - products of derivatives ix, iy(results of sobel operators gx, gy)
    
    for (int i = 0; i< 3; i++){
        i_arr[i] = new long int[n]; 
        for (int j = 0; j< n; j++){
            i_arr[i][j] = i_arr2[i * n + j];
        }
    }
    delete[] img_grayscale2;
    // printf("gaussian\n");
    int mask[25] = {
        1,  4,  7,  4, 1,
        4, 16, 26, 16, 4,
        7, 26, 41, 26, 7,
        4, 16, 26, 16, 4,
        1,  4,  7,  4, 1
    };
    cudaMemcpyToSymbol(cmask, mask, 25 * sizeof(int));
    sum = 0;
    for (int i = 0; i < 3; i++){
        long int* img_gaussian = gaussian_filter(i_arr3[i], width, height); // gaussian filter for ixix iyiy ixiy
        delete[] i_arr3[i];
        i_arr3[i] = img_gaussian;
        
        // for (int j =0; j < width * height; j++) if (i == 1)sum+= i_arr[i][j];
    }
    pad_size = 2;
    n_pad = (width + 2*pad_size)*(height + 2*pad_size);
    long int * dev_gaussian, * dev_padded2;
    long int * padded = new long int[3 * n_pad];
    long int* gaussian = new long int[3 * n];
    cudaMalloc((void**) &dev_padded2, 3 * n_pad * sizeof(long int));
    cudaMalloc((void**) &dev_gaussian, 3 * n * sizeof(long int));
    for (int i = 0; i < 3; i++){
        long int* img_padded = reflection_padding(i_arr[i], width, height, pad_size);
        delete[] i_arr[i];
        i_arr[i] = img_padded;
    }
    for (int i = 0; i < 3; i++){
        for (int j = 0; j < n_pad; j++){
            padded[i * n_pad + j] = i_arr[i][j];
        }
    }
    dim3 blockDim(num_blocks, 3);
    cudaMemcpy(dev_padded2, padded, 3 * n_pad * sizeof(long int), cudaMemcpyHostToDevice);
    gaussian_parallel<<<blockDim, THREADS_PER_BLOCK>>>(dev_padded2, width, height, n, n_pad, mask, dev_gaussian);
    cudaMemcpy(gaussian, dev_gaussian, 3 * n * sizeof(long int), cudaMemcpyDeviceToHost);

    sum = 0;
    for (int i = 0; i < 3; i++){
        delete[] i_arr[i];
        i_arr[i] = new long int[n]; 
        for (int j = 0; j< n; j++){
            i_arr[i][j] = gaussian[i * n + j];
            if(i==1)sum+= i_arr[i][j];
            if (i_arr[i][j] != i_arr3[i][j]) printf("%d,%d: %ld vs %ld\n",i, j, i_arr[i][j], i_arr3[i][j]);
        } 
    }
    cudaFree(dev_padded2);

    delete[] padded;
    delete[] gaussian;

    printf(" %lld\n", sum);
    // printf("before response\n");

    int m[4] = {
        0, 2, // ixix ixiy
        2, 1  // ixiy iyiy
    };

    cudaMemcpyToSymbol(cm, m, 4 * sizeof(int));

    long long int* r_arr, * dev_r_arr;
    r_arr = new long long int[n];
    cudaMalloc((void**) &dev_r_arr, n * sizeof(long long int));
    

    // long long int* r_arr = pixel_response(i_arr, width, height, k); // response function (k constant <0.04-0.06>)
    pixel_response_parallel<<<num_blocks, THREADS_PER_BLOCK>>>(dev_gaussian, width, height, n, k, dev_r_arr); // response function (k constant <0.04-0.06>)
    cudaMemcpy(r_arr, dev_r_arr, n * sizeof(long long int), cudaMemcpyDeviceToHost);

    for (int i = 0; i< 3; i++) delete[] i_arr3[i];
    delete[] i_arr3;
    cudaFree(dev_gaussian);
    cudaFree(dev_r_arr);
    // printf("after response\n");
    bool* t_arr = threshold_response(r_arr, width, height, threshold, n_dim, max_corners); // which points on img are corners

    // for (int y = 0; y < height; y++){ // print corners coords
    //     for (int x=0; x < width; x++){
    //         if (t_arr[y * width + x] == true)
    //             printf("(%d, %d)\n", x, y);
    //     }
    // }

    // printf("coloring\n");
    color_corners(img, width, height, t_arr, channels, cross_size);
    for (int i = 0; i < 3; i++){
        delete[] i_arr[i];
    }
    delete[] i_arr;
    delete[] r_arr;
    delete[] t_arr;

    end = std::chrono::system_clock::now();
    std::chrono::duration<double> computation_time = end - start;
    printf("computation time: %lf seconds\n", computation_time.count());
    printf("finished\n");

    // for(int i = 0; i< width * height; i++)
    //     img[i] = img_grayscale[i];
    // stbi_write_png("1.png", width, height, 1, img, width);
    stbi_write_png("3.png", width, height, channels, img, width * channels);
    delete[] img_grayscale;
    
    stbi_image_free(img);
    return;
}

int main(int argc, char* argv[]) {
    float k = 0.04;
    int max_corners = INT_MAX, cross_size = 3;
    if (argc >= 4 && argc <= 7) {
        printf("img: %s\n", argv[1]); // 1 - nazwa wejściowego obrazu 2 - threshold 3 - n_dim (defines size of window where can only be 1 corner); optional:  4  - k (constant), 5 - max corner count, 7 - cross_size (red crossed in saved picture)
        printf("threshold: %s\n", argv[2]); // 1 000 000 000 - limit in most cases
        printf("n_dim: %s\n", argv[3]);
        if (argc >= 5)
            k = atof(argv[4]);
        if (argc >= 6)
            max_corners = atoi(argv[5]);
        if (argc >= 7)
            cross_size = atoi(argv[6]);
        printf("k: %f\n", k); // <0.4; 0.6>
        printf("max_corners: %d\n", max_corners);
        printf("cross_size: %d\n", cross_size); // best 3 per 500 width/height
    } else {
        printf("podaj poprawna liczbe argumentow\n");
        exit(1);
    }
    
    detect_corners_par(argv[1], atoll(argv[2]), atoi(argv[3]), k, max_corners, cross_size);
    // detect_corners_seq(argv[1], atoll(argv[2]), atoi(argv[3]), k, max_corners, cross_size);
    return 0;
}