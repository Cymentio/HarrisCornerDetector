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

int* img_to_grayscale(unsigned char* img, int width, int height, int channels){
    int* img_grayscale = new int[width*height];
    int i;

    if (channels == 1) {
        # pragma omp parallel for private(i)
        for(i=0; i< width*height; i++)
            img_grayscale[i] = img[i];
        return img_grayscale;
    }
    # pragma omp parallel for private(i)
    for (i=0; i<height*width;i++){
        img_grayscale[i] = img[i * channels] * 0.299 + img[i * channels + 1] * 0.587 + img[i * channels+2] * 0.114;
    }

    return img_grayscale;
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

    #pragma omp parallel private(x, y, i) shared(height, width, gx, gy, i_arr, ix, iy)
    {
        #pragma omp for
        for (y = 0; y < height; y++){
            for (x=0; x < width; x++){
                // operator_values(img_padded, width, y, x, gx, gy, 3, 1, ix, iy); // ix iy
                ix[y * width + x] = operator_value(img_padded, width, y, x, gx, 3, 1); // ix
                iy[y * width + x] = operator_value(img_padded, width, y, x, gy, 3, 1); // iy
            }
        }

        #pragma omp single
        {
        for (i = 0; i < 3; i++){
                    i_arr[i] = new long int[width * height];
            }
        }
        #pragma omp barrier

        #pragma omp for
        for (y = 0; y < height; y++){
            for (x=0; x < width; x++){ // after loop above
                i_arr[0][y * width + x] = ix[y * width + x] * ix[y * width + x]; // ixix
                i_arr[1][y * width + x] = iy[y * width + x] * iy[y * width + x]; // iyiy
                i_arr[2][y * width + x] = ix[y * width + x] * iy[y * width + x]; // ixiy
            }
        }
    }
    delete[] ix;
    delete[] iy;

    delete[] img_padded;
    return i_arr;
}

long int* gaussian_filter(long int* img, int width, int height){ // does convolution on image using mask
    int mask[25] = {
        1,  4,  7,  4, 1,
        4, 16, 26, 16, 4,
        7, 26, 41, 26, 7,
        4, 16, 26, 16, 4,
        1,  4,  7,  4, 1
    };
    long int* img_padded = reflection_padding(img, width, height, 2);
    long int* img_gaussian = new long int[width*height];
    int x, y;

    # pragma omp parallel for collapse(2) private(y, x)
    for (y = 0; y < height; y++){
        for (x = 0; x < width; x++){
            img_gaussian[y * width + x] = gaussian_value(img_padded, width, y, x, mask, 5, 2);
        }
    }

    delete[] img_padded;
    return img_gaussian;
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

    #pragma omp parallel for collapse(2) shared(height, width, i_arr, r_arr) private(y, x, det, tr)
    for (y = 0; y < height; y++){
        for (x=0; x < width; x++){
            det = i_arr[m[0]][y * width + x] * i_arr[m[3]][y * width + x] - i_arr[m[1]][y * width + x] * i_arr[m[2]][y * width + x];
            tr = i_arr[m[0]][y * width + x] + i_arr[m[3]][y * width + x];
            r_arr[y * width + x] = (long long) (det - k * tr * tr);             
        }
    }

    return r_arr;
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

    if(x_step * y_step > 200){ // poniżej nieoptymalnie
        for (y = 0; y <= height - y_step; y+= y_step){
            for (x=0; x <= width - x_step; x+= x_step){
                long long x_mean = 0;
                long long y_mean = 0;
                long long sum_weights = 0;
                long long sum_response = 0;
                #pragma omp parallel for default(shared) collapse(2) reduction(+:sum_weights, x_mean, y_mean, sum_response) private(x1, y1, r)
                for (y1 = y; y1 < y + y_step; y1++){
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
    } else {
        for (y = 0; y <= height - y_step; y+= y_step){
            for (x=0; x <= width - x_step; x+= x_step){
                long long x_mean = 0;
                long long y_mean = 0;
                long long sum_weights = 0;
                long long sum_response = 0;
                for (y1 = y; y1 < y + y_step; y1++){
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
    }

    // leave max_corners corners with highest sum of values
    std::sort(corners.begin(), corners.end(), compare_corner);
    if(corners.size()>=max_corners) corners.resize(max_corners);
    for (auto c: corners){
        t_arr[c.index] = true;
    }

    return t_arr;
}

void color_corners(unsigned char* img, int width, int height, bool* t_arr, int channels, int corner_size){ 
    // adds red crosses to corners
    int x, y;

    #pragma omp parallel for collapse(2)
    for (y = 0; y < height; y++){
        for (x=0; x < width; x++){
            if (t_arr[y * width + x] == true){ // if corner add cross
                if (channels == 1) { // if greyscale black corners
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


    // printf("greyscale\n");
    img_grayscale = img_to_grayscale(img, width, height, channels); // compute grayscale from img

    // printf("derivatives\n");
    i_arr = compute_derivatives(img_grayscale, width, height); // ixix iyiy ixiy - products of derivatives ix, iy(results of sobel operators gx, gy)

    long long sum = 0;
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
    
    detect_corners_seq(argv[1], atoll(argv[2]), atoi(argv[3]), k, max_corners, cross_size);
    return 0;
}