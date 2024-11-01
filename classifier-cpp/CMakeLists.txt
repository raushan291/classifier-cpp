cmake_minimum_required(VERSION 3.10)
project( Infer_OpenCV )

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set the path to OpenCV
set(OpenCV_DIR ./opencv/build)

# Find OpenCV package
find_package( OpenCV REQUIRED )

# Find CUDA package
# find_package(CUDA REQUIRED)

# Include directories
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(${CUDA_INCLUDE_DIRS})

# Add executable
add_executable( Infer_OpenCV ../infer_opencv.cpp )

# Link libraries
target_link_libraries( 
    Infer_OpenCV ${OpenCV_LIBS}
    # ${CUDA_LIBRARIES}
)
