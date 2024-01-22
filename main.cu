#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


__global__ void toGrayscaleCUDA(unsigned char* img, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        unsigned char* pixel = img + (y * width + x) * channels;
        unsigned char gray = (unsigned char)(0.299f * pixel[0] + 0.587f * pixel[1] + 0.114f * pixel[2]);
        pixel[0] = pixel[1] = pixel[2] = gray;
    }
}

__global__ void applySobelCUDA(unsigned char* img, unsigned char* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
        int gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
        int sumX = 0, sumY = 0;

        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                unsigned char* pixel = img + ((y + ky) * width + (x + kx)) * channels;
                sumX += gx[ky + 1][kx + 1] * pixel[0];
                sumY += gy[ky + 1][kx + 1] * pixel[0];
            }
        }

        int magnitude = (int)sqrtf(sumX * sumX + sumY * sumY);
        magnitude = magnitude > 255 ? 255 : magnitude;

        unsigned char* outPixel = output + (y * width + x) * channels;
        for (int c = 0; c < channels; c++) {
            outPixel[c] = (unsigned char)magnitude;
        }
    }
}

void applySobel(unsigned char* img, unsigned char* output, int width, int height, int channels) {
    int gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            int sumX = 0, sumY = 0;
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    unsigned char* pixel = img + ((y + ky) * width + (x + kx)) * channels;
                    sumX += gx[ky + 1][kx + 1] * pixel[0];
                    sumY += gy[ky + 1][kx + 1] * pixel[0];
                }
            }
            int magnitude = (int)sqrt(sumX * sumX + sumY * sumY);
            magnitude = magnitude > 255 ? 255 : magnitude;
            output[(y * width + x) * channels] = (unsigned char)magnitude;
            for (int c = 0; c < channels; c++) {
                output[(y * width + x) * channels + c] = (unsigned char)magnitude;
            }
        }
    }
}


int main()
{
  int width, height, channels;
  unsigned char *img = stbi_load("example.jpg", &width, &height, &channels, 0);
  if (img == nullptr) {
      return -1;
  }

  dim3 dimBlock(16, 16);
  dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);

  unsigned char *dev_img, *dev_output;
  cudaMalloc((void**)&dev_img, width * height * channels);
  cudaMalloc((void**)&dev_output, width * height * channels);

  cudaMemcpy(dev_img, img, width * height * channels, cudaMemcpyHostToDevice);


  toGrayscaleCUDA<<<dimGrid, dimBlock>>>(dev_img, width, height, channels);
  cudaDeviceSynchronize(); 

  applySobelCUDA<<<dimGrid, dimBlock>>>(dev_img, dev_output, width, height, channels);
  cudaDeviceSynchronize(); 


  unsigned char *output = new unsigned char[width * height * channels];
  cudaMemcpy(output, dev_output, width * height * channels, cudaMemcpyDeviceToHost);

  stbi_write_jpg("ścieżka_zapisu.jpg", width, height, channels, output, 100);

 
  stbi_image_free(img);
  cudaFree(dev_img);
  cudaFree(dev_output);
  delete[] output;



    return 0;
}