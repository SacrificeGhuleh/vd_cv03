#include <iostream>
#include <opencv2/opencv.hpp>

#include <random>
#include <functional>
#include <cstdint>
#include "colors.h"


std::random_device randomDevice;
typedef std::mt19937 mt19937Engine;

typedef std::uniform_real_distribution<double> uniformDistribution;
auto uniformGenerator = std::bind(uniformDistribution(0.0f, 1.0f), mt19937Engine(randomDevice()));

typedef std::normal_distribution<double> normalDistribution;

const uint64_t SAMPLES = 300;


const uint64_t X_DOMAIN = 500;
const uint64_t Y_DOMAIN = 500;

const uint64_t X_CENTER = X_DOMAIN / 2;
const uint64_t Y_CENTER = Y_DOMAIN / 2;

const float deviation = 100.f;
const int radius = 100.0f;


float gauss(float center, float deviation, float x) {
  return exp(-((x - center) * (x - center) / (2.f * deviation * deviation)));
}

void debugDrawPointsWithRadius(cv::Mat &mat, const std::vector<cv::Point3i> &points) {
  static int colorCounter = 0;
  for (const auto &point : points) {
//    mat.at<uint8_t>(point.y, point.x) = point.z;
    cv::circle(mat, cv::Point(point.x, point.y), 1, colors[colorCounter], cv::FILLED, cv::LINE_AA);
    cv::circle(mat, cv::Point(point.x, point.y), radius, colors[colorCounter], 1, cv::LINE_AA);
    
    colorCounter++;
    colorCounter %= colorsSize;
  }
}

void debugDrawPointsWithFnVal(cv::Mat &mat, const std::vector<cv::Point3i> &points) {
  for (const auto &point : points) {
//    mat.at<uint8_t>(point.y, point.x) = point.z;
    cv::circle(mat, cv::Point(point.x, point.y), 3, point.z, cv::FILLED, cv::LINE_AA);
  }
  
}


int main(int argc, const char **argv) {
  cv::Mat1b mat = cv::Mat1b::zeros(Y_DOMAIN, X_DOMAIN);
  cv::Mat3b rgbmat = cv::Mat3b::zeros(Y_DOMAIN, X_DOMAIN);
  
  std::vector<cv::Point3i> points;
  
  auto samplePoints = cv::Mat1f(SAMPLES, 2);
  auto samplePointsValues = cv::Mat1f(SAMPLES, 1);
  
  {
    auto normalGeneratorX = std::bind(normalDistribution(X_CENTER, deviation), mt19937Engine(randomDevice()));
    auto normalGeneratorY = std::bind(normalDistribution(Y_CENTER, deviation), mt19937Engine(randomDevice()));
    int x, y, z;
    
    points.reserve(SAMPLES);
    
    for (uint64_t i = 0; i < SAMPLES; i++) {
      x = normalGeneratorX();
      y = normalGeneratorY();
      
      z = gauss(X_CENTER, deviation, x) * gauss(Y_CENTER, deviation, y) * 255;
      if (x < 0.f || x > mat.cols - 1.f || y < 0.f || y > mat.rows - 1.f) {
        --i;
      } else {
        points.emplace_back(cv::Point3f(x, y, z));
        
        samplePoints.at<float>(i, 0) = x;
        samplePoints.at<float>(i, 1) = y;
        samplePointsValues.at<float>(i) = z;
      }
    }
  }

//  debugDrawPointsWithRadius(rgbmat, points);
  debugDrawPointsWithFnVal(mat, points);
  applyColorMap(mat, rgbmat, cv::COLORMAP_HOT);
  
  
  cv::imshow("mat", mat);
  cv::imshow("rgbMat", rgbmat);
  
  cv::waitKey();
  
  cv::Mat1b reconstructionMat = cv::Mat1b::zeros(Y_DOMAIN, X_DOMAIN);
  cv::Mat3b reconstructionRgbMat = cv::Mat3b::zeros(Y_DOMAIN, X_DOMAIN);
  
  
  cv::flann::Index flannIndex = cv::flann::Index(samplePoints, cv::flann::KDTreeIndexParams(1), cvflann::FLANN_DIST_EUCLIDEAN);

//  std::cout << "Sample points: " << samplePoints << std::endl;
//  std::cout << "Sample values: " << samplePointsValues << std::endl;
  
  for (int row = 0; row < mat.rows; row++) {
    for (int col = 0; col < mat.cols; col++) {
      std::vector<int> indices;
      std::vector<float> distances;
      std::vector<float> query = {static_cast<float>(col), static_cast<float>(row)};
      
      
      int n = flannIndex.radiusSearch(query, indices, distances, radius * radius, 10, cv::flann::SearchParams(32));

//      std::cout << "Found " << n << " indices" << std::endl;
      if (n > 0) {
        const int indicesCount = std::min<int>(indices.size(), n);
        
        cv::Point3i nearestNeighbor = cv::Point3i(
            samplePoints.at<float>(0, 0),
            samplePoints.at<float>(0, 1),
            samplePointsValues.at<float>(0));
        float nearestDistance = distances.at(0);
        
        float shepardNumer = 0;
        float shepardDenom = 0;
//        rgbmat.at<cv::Vec3b>(row, col) = colors[n]; // debug
        for (int i = 0; i < indicesCount; i++) {
          const int indice = indices.at(i);
          const float distance = distances.at(i);
          if (distance > 0) {
            const int x = samplePoints.at<float>(indice, 0);
            const int y = samplePoints.at<float>(indice, 1);
            const int z = samplePointsValues.at<float>(indice);
            const cv::Point3i neighbor = cv::Point3i(x, y, z);
            float lambda = static_cast<float>(neighbor.z) / (nearestDistance * radius + nearestDistance);
            lambda *= lambda;
            
            shepardNumer += neighbor.z * exp(-lambda);
            shepardDenom += exp(-lambda);
          }
        }
        reconstructionMat.at<uint8_t>(row, col) = (shepardNumer / shepardDenom);
      }
    }
  }
  
  for (uint64_t i = 0; i < SAMPLES; i++) {
    float x = samplePoints.at<float>(i, 0);
    float y = samplePoints.at<float>(i, 1);
    float z = samplePointsValues.at<float>(i);
    reconstructionMat.at<uint8_t>(y, x) = z;
  }
  
  
  cv::imshow("reconstructionMat", reconstructionMat);
  applyColorMap(reconstructionMat, reconstructionRgbMat, cv::COLORMAP_HOT);
  cv::imshow("reconstructionRgbMat", reconstructionRgbMat);
  
  cv::waitKey();
  return 0;
}