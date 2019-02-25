#!/bin/sh

#C++ compiler
CXX_COMPILER="g++"

# C++ flags
CXX_FLAGS="-O3 -flto -Wall -Wno-unused-function"

# add coverage flags
USE_COVERAGE=0

# use MPI for integration (set this to the same value as in VMC++!)
USE_MPI=0

# MCIntegrator++ Library
MCI_ROOT="/...../MCIntegratorPlusPlus"

# NoisyFunctionMinimization Library
NFM_ROOT="/...../NoisyFunMin"

# VariationalMonteCarlo Library
VMC_ROOT="/...../VMCPlusPlus"

# FeedForwardNeuralNetwork Library
FFNN_ROOT="/...../FeedForwardNeuralNetwork"

# GNU Scientific Library
GSL_ROOT="" # provide a path if not in system location
