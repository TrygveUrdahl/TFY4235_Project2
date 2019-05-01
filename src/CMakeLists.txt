cmake_minimum_required(VERSION 3.1)

project(TFY4235_Project2 CXX)

add_compile_options(-std=c++17 -O3)

find_package(OpenMP)
find_package(Armadillo REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lsuperlu")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -larmadillo -openmp")

file(GLOB project2_SRC
    "*.cpp"
)

include_directories(${ARMADILLO_INCLUDE_DIRS})
add_executable(project2 ${project2_SRC})

if(OpenMP_CXX_FOUND)
    target_link_libraries(project2 PUBLIC OpenMP::OpenMP_CXX)
endif()
