#The name of the project
PROJECT(Matrix_Multiplication_CUDA)

#The CMake Minimum version that is required. The FindCUDA script
#is distributed since version 2.8
#Emerald only has 2.6.3, so we'll have to bootstrap it up
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)


#Searching CUDA
FIND_PACKAGE(CUDA)

#Find MPI
FIND_PACKAGE(MPI)


include_directories (${MPI_INCLUDE_PATH})
#Include the FindCUDA script
INCLUDE(FindCUDA)
INCLUDE(FindMPI)

set(CUDA_NVCC_FLAGS -arch sm_20)
CUDA_ADD_EXECUTABLE(monte_pi monte_pi.cu)
target_link_libraries (monte_pi ${MPI_LIBRARIES})
INSTALL(TARGETS monte_pi RUNTIME DESTINATION bin)