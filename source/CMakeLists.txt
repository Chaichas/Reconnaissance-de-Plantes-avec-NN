    cmake_minimum_required(VERSION 3.10)

# set the project name
project(cnn)
    find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE-DIRECTORIES})
# enable modern C++
    set(CMAKE_CXX_STANDARD 17)
enable_testing()
# add the executable
    add_executable( main output.cpp data.cpp Convolution_layer.cpp Direction_file.cpp Pooling_layer.cpp Random_weights.cpp softmax_layer.cpp)
    target_link_libraries( dspimg ${OpenCV_LIBS})

