#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>

__global__ void toGrayscaleCUDA(unsigned char* img, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        unsigned char* pixel = img + (y * width + x) * channels;
        unsigned char gray = (unsigned char)(0.299f * pixel[0] + 0.587f * pixel[1] + 0.114f * pixel[2]);
        pixel[0] = pixel[1] = pixel[2] = gray;
    }
}

void toGrayscale(unsigned char* img, int width, int height, int channels) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            unsigned char* pixel = img + (y * width + x) * channels;
            unsigned char gray = static_cast<unsigned char>(0.299f * pixel[0] + 0.587f * pixel[1] + 0.114f * pixel[2]);
            pixel[0] = pixel[1] = pixel[2] = gray;
        }
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


void applyBlur(unsigned char* img, unsigned char* output, int width, int height, int channels) {

    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            unsigned int sum[3] = {0,0,0};

            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int newY = y + ky;
                    int newX = x + kx;

                    
                    if (newY >= 0 && newY < height && newX >= 0 && newX < width) {
                        unsigned char* pixel = img + (newY * width + newX) * channels;
                        for (int c = 0; c < channels; c++) {
                            sum[c] += pixel[c];
                        }
                    }
                }
            }

            unsigned char* outPixel = output + (y * width + x) * channels;
            for (int c = 0; c < channels; c++) {
                  outPixel[c] = (unsigned char) (sum[c] / 9);
            }
        }
    }
}

__global__ void applyBlurCUDA(unsigned char* img, unsigned char* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && y >= 1 && x < width - 1 && y < height - 1) {
        unsigned int sum[3] = {0, 0, 0};

        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int newY = y + ky;
                int newX = x + kx;

                unsigned char* pixel = img + (newY * width + newX) * channels;
                for (int c = 0; c < channels; c++) {
                    sum[c] += pixel[c];
                }
            }
        }

        unsigned char* outPixel = output + (y * width + x) * channels;
        for (int c = 0; c < channels; c++) {
            outPixel[c] = (unsigned char)(sum[c] / 9);
        }
    }
}








int main()
{
    while (true) {
        std::cout << "MENU" << std::endl;
        std::cout << "1. Sobel Operator" << std::endl;
        std::cout << "2. Blur" << std::endl;
        std::cout << "3. Convert to grayscale" << std::endl;
        std::cout << "4. Wyjście" << std::endl;
        std::cout << "Wybierz opcję: ";

        int choice;
        std::cin >> choice;
        
        if(choice==4){
          return 0;
        }
        
        std::string input_file_name;
        std::cout << "Input file name: " << std::endl;
        std::cin >> input_file_name;

        std::string output_file_name;
        std::cout << "Output file name: " << std::endl;
        std::cin >> output_file_name;

        int width, height, channels_in_file;
        int channels = 3;
        unsigned char* img = stbi_load(input_file_name.c_str(), &width, &height, &channels_in_file, channels);

        unsigned char* output = new unsigned char[width * height * channels];
        unsigned char* cpu_output = new unsigned char[width * height * channels];
        unsigned char *dev_img, *dev_output;
        cudaMalloc(&dev_img, width * height * channels);
        cudaMalloc(&dev_output, width * height * channels);

        cudaMemcpy(dev_img, img, width * height * channels, cudaMemcpyHostToDevice);

        dim3 dimBlock(16, 16);
        dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
          
        std::chrono::high_resolution_clock::time_point cpu_start,cpu_stop,gpu_start,gpu_stop;

        switch (choice) {
            case 1:
                gpu_start = std::chrono::high_resolution_clock::now();
                toGrayscaleCUDA<<<dimGrid, dimBlock>>>(dev_img, width, height, channels);
                applySobelCUDA<<<dimGrid, dimBlock>>>(dev_img, dev_output, width, height, channels);
                gpu_stop = std::chrono::high_resolution_clock::now();
                cpu_start = std::chrono::high_resolution_clock::now();
                toGrayscale(img,width,height,channels);
                applySobel(img,cpu_output,width,height,channels);
                cpu_stop = std::chrono::high_resolution_clock::now();
                break;
            case 2:
                gpu_start = std::chrono::high_resolution_clock::now();
                applyBlurCUDA<<<dimGrid, dimBlock>>>(dev_img, dev_output, width, height, channels);
                gpu_stop = std::chrono::high_resolution_clock::now();
                cpu_start = std::chrono::high_resolution_clock::now();
                applyBlur(img,cpu_output,width,height,channels);
                cpu_stop = std::chrono::high_resolution_clock::now();
                break;
            case 3:
                gpu_start = std::chrono::high_resolution_clock::now();
                toGrayscaleCUDA<<<dimGrid, dimBlock>>>(dev_img,width, height, channels);
                gpu_stop = std::chrono::high_resolution_clock::now();
                dev_output = dev_img;
                cpu_start = std::chrono::high_resolution_clock::now();
                toGrayscale(img,width,height,channels);
                cpu_stop = std::chrono::high_resolution_clock::now();
                break;
            case 4:
                std::cout << "Wyjście z programu." << std::endl;
                return 0;
            default:
                std::cout << "Nieprawidłowa opcja. Spróbuj ponownie." << std::endl;
                break;
        }

        std::chrono::microseconds duration_cpu = std::chrono::duration_cast<std::chrono::microseconds>(cpu_stop - cpu_start);
        std::chrono::microseconds duration_gpu = std::chrono::duration_cast<std::chrono::microseconds>(gpu_stop - gpu_start);
        std::cout << "Cpu time: " << duration_cpu.count() << " microseconds" << std::endl;
        std::cout << "Gpu time: " << duration_gpu.count() << " microseconds" << std::endl;
        cudaMemcpy(output, dev_output, width * height * channels, cudaMemcpyDeviceToHost);

        stbi_write_jpg(output_file_name.c_str(), width, height, channels, output, 100);

        stbi_image_free(img);
        cudaFree(dev_img);
        cudaFree(dev_output);
        delete[] output;
    }



    return 0;
}