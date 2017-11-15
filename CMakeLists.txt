cmake_minimum_required(VERSION 2.8.3)
project(disp)



find_package(OpenCV REQUIRED)



include_directories(${catkin_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

link_directories(${OpenCV_LIBRARY_DIRS})

add_definitions(-std=c++0x)

add_executable(disparity src/src.cpp)

target_link_libraries(disparity
	${catkin_LIBRARIES}
        ${OpenCV_LIBRARIES}
)
