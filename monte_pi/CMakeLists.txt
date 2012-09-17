#The name of the project
PROJECT(Matrix_Multiplication_CUDA)

#The CMake Minimum version that is required. The FindCUDA script
#is distributed since version 2.8
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)


#Searching CUDA
FIND_PACKAGE(CUDA)



#Include the FindCUDA script
INCLUDE(FindCUDA)

#The rule to build the executable program matrixMul from matrixmul.cu
# and matrixMul_gold.cpp
CUDA_ADD_EXECUTABLE(monte_pi monte_pi.cu)