#ifndef test_H
#define test_H

class HarrisCornerDetector {
public:
    virtual void start(){};
    void setImgPath(const char* path);
    ~HarrisCornerDetector();
protected:
    int width;
    int height;
    int channels;
    virtual unsigned char* imgToGrayscale(unsigned char* img){return nullptr;};
    const char* img_path;
};

void detectCorners(HarrisCornerDetector* t);

class SeqHarrisCornerDetector : public HarrisCornerDetector {
public:
    SeqHarrisCornerDetector();
    void start();
private:
    unsigned char* imgToGrayscale(unsigned char* img);
};

class ParHarrisCornerDetector : public HarrisCornerDetector {
public:
    ParHarrisCornerDetector(unsigned int n, unsigned int d);
    void start();
    __device__ void dev();
private:
    unsigned int blocks_n; // liczba bloków
    unsigned int block_dim; // rozmiar bloku
    unsigned char* imgToGrayscale(unsigned char* img);
};

__global__ void aaa(ParHarrisCornerDetector* h);

#endif