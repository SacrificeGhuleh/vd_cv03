CMAKE_MINIMUM_REQUIRED(VERSION 3.14)
PROJECT(CV03)

# add libs you need
SET(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs opencv_calib3d)
SET(SRC main.cpp)

ADD_EXECUTABLE(${PROJECT_NAME} ${SRC} cli.h colors.h)

#find_package(OpenMP REQUIRED)

#if (OpenMP_CXX_FOUND)
#  target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} OpenMP::OpenMP_CXX)
#else ()
  target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})
#endif ()

