#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>


__global__ void resizeImageCUDA(unsigned char* input, unsigned char* output, int inWidth, int inHeight, int channels, int outWidth, int outHeight) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < outWidth && y < outHeight) {
        float scaleX = static_cast<float>(inWidth) / static_cast<float>(outWidth);
        float scaleY = static_cast<float>(inHeight) / static_cast<float>(outHeight);

        float srcX = static_cast<float>(x) * scaleX;
        float srcY = static_cast<float>(y) * scaleY;

        int x0 = static_cast<int>(srcX);
        int y0 = static_cast<int>(srcY);

        int x1 = x0 + 1;
        int y1 = y0 + 1;

        float weightX = srcX - x0;
        float weightY = srcY - y0;

        for (int c = 0; c < channels; ++c) {
            float interpolatedValue = (1 - weightX) * (1 - weightY) * input[(y0 * inWidth + x0) * channels + c] +
                weightX * (1 - weightY) * input[(y0 * inWidth + x1) * channels + c] +
                (1 - weightX) * weightY * input[(y1 * inWidth + x0) * channels + c] +
                weightX * weightY * input[(y1 * inWidth + x1) * channels + c];

            output[(y * outWidth + x) * channels + c] = static_cast<unsigned char>(interpolatedValue);
        }
    }
}

void resizeImageCPU(unsigned char* input, unsigned char* output, int inWidth, int inHeight, int channels, int outWidth, int outHeight) {
    for (int y = 0; y < outHeight; ++y) {
        for (int x = 0; x < outWidth; ++x) {
            float scaleX = static_cast<float>(inWidth) / static_cast<float>(outWidth);
            float scaleY = static_cast<float>(inHeight) / static_cast<float>(outHeight);

            float srcX = static_cast<float>(x) * scaleX;
            float srcY = static_cast<float>(y) * scaleY;

            int x0 = static_cast<int>(srcX);
            int y0 = static_cast<int>(srcY);

            int x1 = x0 + 1;
            int y1 = y0 + 1;

            float weightX = srcX - x0;
            float weightY = srcY - y0;

            for (int c = 0; c < channels; ++c) {
                float interpolatedValue = (1 - weightX) * (1 - weightY) * input[(y0 * inWidth + x0) * channels + c] +
                    weightX * (1 - weightY) * input[(y0 * inWidth + x1) * channels + c] +
                    (1 - weightX) * weightY * input[(y1 * inWidth + x0) * channels + c] +
                    weightX * weightY * input[(y1 * inWidth + x1) * channels + c];

                output[(y * outWidth + x) * channels + c] = static_cast<unsigned char>(interpolatedValue);
            }
        }
    }
}

__global__ void rotate180CUDA(unsigned char* input, unsigned char* output, int width, int height, int channels) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height) {
        int input_index = (y * width + x) * channels;
        int output_index = ((height - 1 - y) * width + (width - 1 - x)) * channels;

        for (int c = 0; c < channels; ++c) {
            output[output_index + c] = input[input_index + c];
        }
    }
}

void rotate180_CPU(unsigned char* input, unsigned char* output, int width, int height, int channels) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int input_index = (y * width + x) * channels;
            int output_index = ((height - 1 - y) * width + (width - 1 - x)) * channels;

            for (int c = 0; c < channels; ++c) {
                output[output_index + c] = input[input_index + c];
            }
        }
    }
}


__global__ void translateImageCUDA(unsigned char* input, unsigned char* output, int width, int height, int channels, int dx) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height) {
        int input_index = (y * width + x) * channels;
        int output_index = ((y)*width + (x + dx)) * channels;

        if (output_index >= 0 && output_index < width * height * channels) {
            for (int c = 0; c < channels; ++c) {
                output[output_index + c] = input[input_index + c];
            }
        }
    }
}


void translateImageCPU(unsigned char* input, unsigned char* output, int width, int height, int channels, int dx) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int input_index = (y * width + x) * channels;
            int output_x = x + dx;
            int output_y = y;

            if (output_x >= 0 && output_x < width && output_y >= 0 && output_y < height) {
                int output_index = (output_y * width + output_x) * channels;

                for (int c = 0; c < channels; ++c) {
                    output[output_index + c] = input[input_index + c];
                }
            }
        }
    }
}

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
        int gx[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
        int gy[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };
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
    int gx[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
    int gy[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };

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
            unsigned int sum[3] = { 0,0,0 };

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
                outPixel[c] = (unsigned char)(sum[c] / 9);
            }
        }
    }
}

__global__ void applyBlurCUDA(unsigned char* img, unsigned char* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && y >= 1 && x < width - 1 && y < height - 1) {
        unsigned int sum[3] = { 0, 0, 0 };

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
        std::cout << "4. Resize" << std::endl;
        std::cout << "5. Rotate 180deg" << std::endl;
        std::cout << "6. Translate" << std::endl;
        std::cout << "7. Exit" << std::endl;
        std::cout << "Choose option: ";

        int choice;
        std::cin >> choice;


        int newWidth, newHeight;
        int dX, dY;

        if (choice == 4) {
            std::cout << "New width: " << std::endl;
            std::cin >> newWidth;
            std::cout << "New height: " << std::endl;
            std::cin >> newHeight;
        }


        if (choice == 6) {
            std::cout << "dx: " << std::endl;
            std::cin >> dX;
        }

        if (choice == 7) {
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



        unsigned char* dev_img, * dev_output;


        dim3 dimBlock(16, 16);
        dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
        unsigned char* cpu_output;
        std::chrono::high_resolution_clock::time_point cpu_start, cpu_stop, gpu_start, gpu_stop;
        gpu_start = std::chrono::high_resolution_clock::now();
        cudaMalloc(&dev_img, width * height * channels);


        cudaMemcpy(dev_img, img, width * height * channels, cudaMemcpyHostToDevice);
        switch (choice) {
        case 1:
            cpu_output = new unsigned char[width * height * channels];
            cudaMalloc(&dev_output, width * height * channels);
            toGrayscaleCUDA << <dimGrid, dimBlock >> > (dev_img, width, height, channels);
            applySobelCUDA << <dimGrid, dimBlock >> > (dev_img, dev_output, width, height, channels);
            cpu_start = std::chrono::high_resolution_clock::now();
            toGrayscale(img, width, height, channels);
            applySobel(img, cpu_output, width, height, channels);
            cpu_stop = std::chrono::high_resolution_clock::now();
            break;
        case 2:
            cpu_output = new unsigned char[width * height * channels];
            cudaMalloc(&dev_output, width * height * channels);
            applyBlurCUDA << <dimGrid, dimBlock >> > (dev_img, dev_output, width, height, channels);
            cpu_start = std::chrono::high_resolution_clock::now();
            applyBlur(img, cpu_output, width, height, channels);
            cpu_stop = std::chrono::high_resolution_clock::now();
            break;
        case 3:
            cpu_output = new unsigned char[width * height * channels];
            cudaMalloc(&dev_output, width * height * channels);
            toGrayscaleCUDA << <dimGrid, dimBlock >> > (dev_img, width, height, channels);
            dev_output = dev_img;
            cpu_start = std::chrono::high_resolution_clock::now();
            toGrayscale(img, width, height, channels);
            cpu_stop = std::chrono::high_resolution_clock::now();
            break;
        case 4:
            cpu_output = new unsigned char[newWidth * newHeight * channels];
            cudaMalloc(&dev_output, newWidth * newHeight * channels);
            resizeImageCUDA << <dimGrid, dimBlock >> > (dev_img, dev_output, width, height, channels, newWidth, newHeight);
            cpu_start = std::chrono::high_resolution_clock::now();
            toGrayscale(img, width, height, channels);
            cpu_stop = std::chrono::high_resolution_clock::now();
            resizeImageCPU(img, cpu_output, width, height, channels, newWidth, newHeight);
            width = newWidth;
            height = newHeight;
            break;
        case 5:
            cpu_output = new unsigned char[width * height * channels];
            cudaMalloc(&dev_output, width * height * channels);
            rotate180CUDA << <dimGrid, dimBlock >> > (dev_img, dev_output, width, height, channels);
            cpu_start = std::chrono::high_resolution_clock::now();
            rotate180_CPU(img, cpu_output, width, height, channels);
            cpu_stop = std::chrono::high_resolution_clock::now();
            break;
        case 6:
            cpu_output = new unsigned char[width * height * channels];
            cudaMalloc(&dev_output, width * height * channels);
            translateImageCUDA << <dimGrid, dimBlock >> > (dev_img, dev_output, width, height, channels, dX);
            cpu_start = std::chrono::high_resolution_clock::now();
            translateImageCPU(img, cpu_output, width, height, channels, dX);
            cpu_stop = std::chrono::high_resolution_clock::now();
            break;
        case 7:
            std::cout << "Wyjście z programu." << std::endl;
            return 0;
        default:
            std::cout << "Nieprawidłowa opcja. Spróbuj ponownie." << std::endl;
            break;
        }
        unsigned char* output = new unsigned char[width * height * channels];
        cudaMemcpy(output, dev_output, width * height * channels, cudaMemcpyDeviceToHost);
        gpu_stop = std::chrono::high_resolution_clock::now();
        std::chrono::microseconds duration_cpu = std::chrono::duration_cast<std::chrono::microseconds>(cpu_stop - cpu_start);
        std::chrono::microseconds duration_gpu = std::chrono::duration_cast<std::chrono::microseconds>(gpu_stop - gpu_start) - duration_cpu;
        std::cout << "Cpu time: " << duration_cpu.count() << " microseconds" << std::endl;
        std::cout << "Gpu time: " << duration_gpu.count() << " microseconds" << std::endl;


        stbi_write_jpg(output_file_name.c_str(), width, height, channels, output, 100);

        stbi_image_free(img);
        cudaFree(dev_img);
        cudaFree(dev_output);
        delete[] output;
    }



    return 0;
}