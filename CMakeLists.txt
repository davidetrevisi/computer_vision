cmake_minimum_required(VERSION 3.17)

project(Final Project)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} Code/include)

add_executable(main Code/src/file.cpp)

target_link_libraries(main ${OpenCV_LIBS})
