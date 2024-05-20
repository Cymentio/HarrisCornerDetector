#include <iostream>
#include <cstdlib>
#include <iostream>
#include <typeinfo>
#include <cuda.h>
#include <windows.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"

int* img_to_grayscale(unsigned char* img, int width, int height, int channels){
    int* img_grayscale = new int[width*height];

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

long int gaussian_value(long int* img, int width, int y, int x, int mask[], int mask_size, int half_mask){
    long long int result = 0;

    for(int y1 = 0; y1 < mask_size; y1++){
        for(int x1 = 0; x1 < mask_size; x1++){
            if ((x + x1) >= (width + mask_size - 1))
                printf("0000000000000000000000");
            result += mask[y1 * mask_size + x1] * img[(y + y1) * (width + mask_size - 1) + (x + x1)];
        }
    }

    // for(int y1 = -half_mask; y1 <= half_mask; y1++){
    //     for(int x1 = -half_mask; x1 <= half_mask; x1++){
    //         result += mask[(y1 + half_mask) * mask_size + (x1 + half_mask)] * img[(y + y1 + half_mask) * (width + mask_size - 1) + (x + x1 + half_mask)];
    //         // mask[(y1 + half_mask) * mask_size + (x1 + half_mask)] - żeby zacząć od 0, 0 w mask
    //     }
    // }

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
            // if (x < padding_size){ // skipped variable for x_diff
            //     img_padded[y * (width + padding_size) + x] = img[(y - padding_size + y_diff) * width - x + padding_size - 1];
            //     continue;
            // }
            // if (x >= width + padding_size){
            //     img_padded[y * (width + padding_size) + x] = img[(y - padding_size + y_diff) * width - x + padding_size + 3];
            //     continue;
            // }
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

    for (int y=0; y < height + 2*padding_size; y++){
        int y_diff = 0;
        if (y < padding_size){
            y_diff = (padding_size - y) * 2; // [hp = padding_size/2] 0 (-hp original) -> hp + 1 (hp - 1 in original); hp - 1 (-1 original) -> hp (0 in original)
        }
        if (y >= height + padding_size){
            y_diff = (height + padding_size - 1 - y) * 2;
        }
        for (int x=0; x < width + 2*padding_size; x++){
            // if (x < padding_size){ // skipped variable for x_diff
            //     img_padded[y * (width + padding_size) + x] = img[(y - padding_size + y_diff) * width - x + padding_size - 1];
            //     continue;
            // }
            // if (x >= width + padding_size){
            //     img_padded[y * (width + padding_size) + x] = img[(y - padding_size + y_diff) * width - x + padding_size + 3];
            //     continue;
            // }
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

    for(int y1 = 0; y1 < mask_size; y1++){
        for(int x1 = 0; x1 < mask_size; x1++){
            result += mask[y1 * mask_size + x1] * img[(y + y1) * (width + mask_size - 1) + (x + x1)];
        }
    }

    // for(int y1 = -half_mask; y1 <= half_mask; y1++){
    //     for(int x1 = -half_mask; x1 <= half_mask; x1++){
    //         result += mask[(y1 + half_mask) * mask_size + (x1 + half_mask)] * img[(y + y1 + half_mask) * (width + mask_size - 1) + (x + x1 + half_mask)];
    //         // mask[(y1 + half_mask) * mask_size + (x1 + half_mask)] - żeby zacząć od 0, 0 w mask
    //     }
    // }

    return result;
}

// does convolution on image using sobel operators and returns products of derivatives (sobel operator results)
long int** compute_derivatives(int* img, int width, int height){ 
    int* ix = new int[width * height];
    int* iy = new int[width * height];
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

    printf("padding\n");
    int* img_padded = reflection_padding(img, width, height, 1);
    printf("padded\n");

    long int** i_arr = new long int*[3]; // ixix iyiy ixiy

    for (int y = 0; y < height; y++){
        for (int x=0; x < width; x++){
            ix[y * width + x] = operator_value(img_padded, width, y, x, gx, 3, 1); // ix
            iy[y * width + x] = operator_value(img_padded, width, y, x, gy, 3, 1); // iy
        }
    }
    for (int i = 0; i < 3; i++){
            i_arr[i] = new long int[width * height];
    }
    for (int y = 0; y < height; y++){
        for (int x=0; x < width; x++){ // after loop above
            i_arr[0][y * width + x] = ix[y * width + x] * ix[y * width + x]; // ixix
            i_arr[1][y * width + x] = iy[y * width + x] * iy[y * width + x]; // iyiy
            i_arr[2][y * width + x] = ix[y * width + x] * iy[y * width + x]; // ixiy
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

    for (int y = 0; y < height; y++){
        for (int x = 0; x < width; x++){
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

    for (int y = 0; y < height; y++){
        for (int x=0; x < width; x++){
            long long det = i_arr[m[0]][y * width + x] * i_arr[m[3]][y * width + x] - i_arr[m[1]][y * width + x] * i_arr[m[2]][y * width + x];
            long long tr = i_arr[m[0]][y * width + x] + i_arr[m[3]][y * width + x];
            r_arr[y * width + x] = (long long) (det - k * tr * tr);             
        }
    }

    return r_arr;
}

bool* threshold_response(long long* r_arr, int width, int height, long long threshold, int n_dim){ // 1 if value above threshold 0 if below for each pixel
    // (2*n_dim + 1) * (2*n_dim + 1) window (neighborhood), only if center pixel is has the highest value in neghborhood it can give 1 as response
    // n_dim - how many pixel from center to each side is neighborhood
    bool* t_arr = new bool[width*height];
    int arr_size = width * height;
    for (int y = 0; y < height; y++){
        for (int x=0; x < width; x++){
            long long r = r_arr[y*width + x];
            if (threshold < r){
                // printf("a ");
                bool not_max_in_neigh = false;
                for (int y1 = -n_dim; y1 <= n_dim; y1++){
                    for (int x1 = -n_dim; x1 <= n_dim; x1++){
                        if ((y + y1)*width + (x + x1) >=0 && (y + y1)*width + (x + x1) < arr_size && // neigbor inside image
                        r_arr[(y + y1)*width + (x + x1)] > r){ // neighbor has higher value
                            not_max_in_neigh = true;
                            break;
                        }
                    }
                    if (not_max_in_neigh) // no need to check other neighboors
                        break;
                }
                if (!not_max_in_neigh){
                    t_arr[y*width + x] = true;
                    continue;
                }   
                // t_arr[y*width + x] = true;
                // continue;                
            }
            t_arr[y*width + x] = false;
        }
    }

    return t_arr;
}

void color_corners(unsigned char* img, int width, int height, bool* t_arr, int channels, int corner_size){ 
    // adds red crosses to corners
    for (int y = 0; y < height; y++){
        for (int x=0; x < width; x++){
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
                    // img_grayscale[i] = img[i * channels] * 0.299 + img[i * channels + 1] * 0.587 + img[i * channels+2] * 0.114;
                }
            }
        }
    }
}

// unsigned char* remove_alpha(unsigned char* img, int width, int height){
//     unsigned char* rgb_img = new unsigned char[width*height*3];
//     printf("ege\n");
//     for (int i=0; i<height*width;i++){
//         for (int c = 0; c < 3; c++){
//             if (i * 3 + c >= 300000)
//                 printf("%d, %d ", i * 3 + c, i * 4 + c);
//             rgb_img[i * 3 + c] = img[i * 4 + c];
//         }
//     }
//     printf("aha\n");
//     delete[] img;
//     return rgb_img;
// }

void detect_corners_seq(const char* img_path, long long threshold, int n_dim, float k){
    int width, height, channels;
    printf("aha\n");
    printf("start sequential\n");
    printf("%s", img_path);
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
    // if(channels == 4){
    //     img = remove_alpha(img, width, height);
    // }

    printf("greyscale\n");
    int* img_grayscale = img_to_grayscale(img, width, height, channels); // compute grayscale from img

    // int* img_g = gaussian_filter((int*)img_grayscale, width, height);
    printf("derivatives\n");
    long int** i_arr = compute_derivatives(img_grayscale, width, height); // ixix iyiy ixiy - products of derivatives ix, iy(results of sobel operators gx, gy)

    printf("gaussian\n");
    for (int i = 0; i < 3; i++){
        long int* img_gaussian = gaussian_filter(i_arr[i], width, height); // gaussian filter for ixix iyiy ixiy
        delete[] i_arr[i];
        i_arr[i] = img_gaussian;
    }
    printf("before response\n");

    long long int* r_arr = pixel_response(i_arr, width, height, k); // response function (k constant <0.04-0.06>)

    long long max = -100000000;
    int max_y = -1;
    int max_x = -1;
    for (int y = 0; y < height; y++){
        for (int x=0; x < width; x++){
            if (r_arr[y * width + x] > max){
                max = r_arr[y * width + x];
                max_y = y;
                max_x = x;
            }
        }
    }
    printf("(%d, %d) : %lld\n", max_x, max_y, max);
    
    printf("after response\n");

    bool* t_arr = threshold_response(r_arr, width, height, threshold, n_dim); // which points on img are corners

    for (int y = 0; y < height; y++){
        for (int x=0; x < width; x++){
            if (t_arr[y * width + x] == true)
                printf("(%d, %d)\n", x, y);
        }
    }

    printf("coloring\n");
    color_corners(img, width, height, t_arr, channels, 3);
    for (int i = 0; i < 3; i++){
        delete[] i_arr[i];
    }
    delete[] i_arr;
    delete[] r_arr;
    delete[] t_arr;
    printf("finished\n");

    // stbi_write_png("1.png", width, height, 1, img_grayscale, width);
    stbi_write_png("2.png", width, height, channels, img, width * channels);
    delete[] img_grayscale;
    
    stbi_image_free(img);
}

void detect_corners_par(){
    printf("start parallel\n");
    // aaa<<<blocks_n, block_dim>>>(this);
    // Sleep(2000);
    // aaa<<<blocks_n, block_dim>>>(this);
    // printf("eee\n");
}

__global__ void aaa(){
    printf("aaaaaaaaaaaaaaaaaaaaaa\n");
}

int main(int argc, char* argv[]) {
    float k = 0.04;
    if (argc == 4 || argc == 5) {
        printf("img: %s\n", argv[1]); // 1 - nazwa wejściowego obrazu 2 - threshold 3 - n_dim (defines size of window where can only be 1 corner) 4 (optional) - k (constant)
        printf("threshold: %s\n", argv[2]);
        printf("n_dim: %s\n", argv[3]);
        if (argc == 5)
            k = atof(argv[4]);
        printf("k: %f\n", k);
    } else {
        printf("podaj poprawna liczbę argumentow: 1\n");
        exit(1);
    }
    printf("aaa\n");
    detect_corners_seq(argv[1], atoll(argv[2]), atoi(argv[3]), k);
    printf("\n");
//  std::clog << "Wersja: "  << CUDA_VERSION << std::endl;
//  std::clog << "Wersja: "  << CUDART_VERSION << std::endl;;
    return 0;
}