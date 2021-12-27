#include <iostream>
#include <algorithm>
#include <chrono>

#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/opencv.hpp>

const int CONVOLUTION_SIZE = 5;

double getCudaSumOfAllPixels(cv::cuda::GpuMat& src) {
  std::vector<cv::cuda::GpuMat> gpuMats;
  cv::cuda::split(src, gpuMats);
  auto redChannel = gpuMats[0];

  return cv::cuda::sum(redChannel).val[0];
}

double getSumOfAllPixels(cv::Mat& src) {
  std::vector<cv::Mat> cpuMats;
  cv::split(src, cpuMats);
  auto redChannel = cpuMats[0];

  return cv::sum(redChannel).val[0];
}

double getCudaMinValue(cv::cuda::GpuMat& src) {
  std::vector<cv::cuda::GpuMat> gpuMats;
  cv::cuda::split(src, gpuMats);
  auto redChannel = gpuMats[0];

  double minValue, maxValue;
  cv::cuda::minMax(redChannel, &minValue, &maxValue);
  return minValue;
}

double getCpuMinValue(cv::Mat& src) {
  std::vector<cv::Mat> cpuMats;
  cv::split(src, cpuMats);
  auto redChannel = cpuMats[0];

  double minValue, maxValue;
  cv::minMaxLoc(redChannel, &minValue, &maxValue);
  return minValue;
}

void getCpuConvolution(cv::Mat& src, cv::Mat& kernel, cv::Mat& convoledImage) {
  std::vector<cv::Mat> cpuMats;
  std::vector<cv::Mat> convoledMats;
  cv::split(src, cpuMats);

  for (auto& channel : cpuMats) {
    cv::Mat convoledChanel;
    cv::filter2D(channel, convoledChanel, -1, kernel, cv::Point(-1, -1), 0, 4);
    convoledMats.push_back(convoledChanel);
  }

  cv::merge(convoledMats, convoledImage);
}

void getCudaConvolution(cv::cuda::GpuMat& src, cv::cuda::GpuMat& kernel, cv::cuda::GpuMat& convoledImage) {
  std::vector<cv::cuda::GpuMat> gpuMats, convoledMats;
  cv::cuda::split(src, gpuMats);

  cv::Ptr<cv::cuda::Convolution> convolver = cv::cuda::createConvolution(cv::Size(CONVOLUTION_SIZE, CONVOLUTION_SIZE));

  for (auto& channel : gpuMats) {
    cv::cuda::GpuMat convoledChanel;
    channel.convertTo(channel, CV_32FC1, 1.0 / 255.0);
    convolver->convolve(channel, kernel, convoledChanel);
    convoledMats.push_back(convoledChanel);
  }

  cv::cuda::merge(convoledMats, convoledImage);
}

int main()
{
  // cv::cuda::printCudaDeviceInfo(0);
  cv::Mat img = cv::imread("palm.jpg", cv::IMREAD_COLOR);
  // cv::imshow("image", img);

  cv::cuda::GpuMat src;
  src.upload(img);

  // sum of pixels
  auto start = std::chrono::high_resolution_clock::now();
  double sumOfAllPixels = getCudaSumOfAllPixels(src);
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  auto cudaCountPixelsDuration = duration.count();

  std::cout << std::fixed << "Sum of all pixels: " << sumOfAllPixels << " on CUDA took: " << cudaCountPixelsDuration << " microseconds" << std::endl;

  start = std::chrono::high_resolution_clock::now();
  sumOfAllPixels = getSumOfAllPixels(img);
  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  auto cpuCountPixelsDuration = duration.count();

  std::cout << std::fixed << "Sum of all pixels: " << sumOfAllPixels << " on CPU took: " << cpuCountPixelsDuration << " microseconds" << std::endl;
  std::cout << "CUDA faster by " << 100 - ((double)cudaCountPixelsDuration / (double)cpuCountPixelsDuration * 100.0) << " %" << std::endl << std::endl;

  // min pixel
  start = std::chrono::high_resolution_clock::now();
  double minCudaValue = getCudaMinValue(src);
  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  auto cudaMinPixelsDuration = duration.count();

  std::cout << std::fixed << "Min pixel: " << minCudaValue << " on CUDA took: " << cudaMinPixelsDuration << " microseconds" << std::endl;

  start = std::chrono::high_resolution_clock::now();
  double minCpuValue = getCpuMinValue(img);
  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  auto cpuMinPixelsDuration = duration.count();

  std::cout << std::fixed << "Min pixel: " << minCpuValue << " on CPU took: " << cpuMinPixelsDuration << " microseconds" << std::endl;
  std::cout << "CUDA faster by " << 100 - ((double)cudaMinPixelsDuration / (double)cpuMinPixelsDuration * 100.0) << " %" << std::endl << std::endl;

  // convolution

  cv::Mat kernel = cv::Mat::ones(CONVOLUTION_SIZE, CONVOLUTION_SIZE, CV_32FC1);
  kernel = kernel / 25;
  cv::cuda::GpuMat gpuKernel, gpuConvolutionOut;
  gpuKernel.upload(kernel);

  cv::Mat blurredImg, blurredImgCuda;

  start = std::chrono::high_resolution_clock::now();
  getCudaConvolution(src, gpuKernel, gpuConvolutionOut);
  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  auto cudaChannelsConvolutionsDuration = duration.count();
  gpuConvolutionOut.download(blurredImgCuda);

  std::cout << std::fixed << "RGB channel convolution on CUDA took: " << cudaChannelsConvolutionsDuration << " microseconds" << std::endl;

  start = std::chrono::high_resolution_clock::now();
  getCpuConvolution(img, kernel, blurredImg);
  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  auto cpuChannelsConvolutionsDuration = duration.count();

  std::cout << std::fixed << "RGB channel convolution on CPU took: " << cpuChannelsConvolutionsDuration << " microseconds" << std::endl;
  std::cout << "CUDA faster by " << 100 - ((double)cudaChannelsConvolutionsDuration / (double)cpuChannelsConvolutionsDuration * 100.0) << " %" << std::endl << std::endl;

  cv::imshow("Original", img);
  cv::imshow("Kernel blur CPU", blurredImg);
  cv::imshow("Kernel blur CUDA", blurredImgCuda);

  cv::waitKey(0);
  return 0;
}
