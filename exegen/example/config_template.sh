#!/bin/sh

#C++ compiler
CXX_COMPILER="g++"

# C++ flags
CXX_FLAGS="-O3 -flto -Wall -Wno-unused-function"

# use MPI for integration (set this to the same value that VMC++/NNVMC were compiled with!)
USE_MPI=0

# MCIntegrator++ Library
MCI_ROOT="/...../MCIntegratorPlusPlus"

# NoisyFunctionMinimization Library
NFM_ROOT="/...../NoisyFunMin"

# VariationalMonteCarlo Library
VMC_ROOT="/...../VMCPlusPlus"

# QNets Library
QNETS_ROOT="/...../QNets"

# sannifa Library
SANNIFA_ROOT="/...../sannifa"

# NNVMC Library
NNVMC_ROOT="/...../NNVMC"

# GNU Scientific Library
GSL_ROOT="" # provide a path if not in system location
