cmake_minimum_required(VERSION 3.17)

project(Final_Project)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} Code/include)

add_executable(main Code/src/main.cpp Code/src/evaluation_metrics.cpp Code/src/player.cpp Code/src/bounding_box.cpp)

target_link_libraries(main ${OpenCV_LIBS})
