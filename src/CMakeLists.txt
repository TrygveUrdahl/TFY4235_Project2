cmake_minimum_required(VERSION 3.1)

project(TFY4235_Project2 CXX)

add_compile_options(-std=c++11 -O3)

find_package(OpenMP)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -larmadillo")

add_executable(project2 main.cpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(project2 PUBLIC OpenMP::OpenMP_CXX)
endif()
