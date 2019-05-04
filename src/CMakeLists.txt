cmake_minimum_required(VERSION 3.1)

project(TFY4235_Project2 CXX C)

add_compile_options(-std=c++17 -O3)

find_package(OpenMP)
find_package(Armadillo REQUIRED)
find_package(HDF5)


# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lsuperlu")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -larmadillo -openmp")

file(GLOB project2_SRC
    "*.cpp"
)

include_directories(${ARMADILLO_INCLUDE_DIRS} ${HDF5_INCLUDE_DIR})
set(_hdf5_libs hdf5 hdf5_cpp)
add_executable(project2 ${project2_SRC})
target_link_libraries(project2 PUBLIC ${_hdf5_libs})

if(OpenMP_CXX_FOUND)
    target_link_libraries(project2 PUBLIC OpenMP::OpenMP_CXX)
endif()
