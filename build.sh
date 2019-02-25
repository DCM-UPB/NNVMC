#!/bin/sh

. ./config.sh
mkdir -p build && cd build
cmake -DCMAKE_CXX_COMPILER="${CXX_COMPILER}" -DUSER_CXX_FLAGS="${CXX_FLAGS}" -DUSE_COVERAGE="${USE_COVERAGE}" -DUSE_MPI="${USE_MPI}" -DMCI_ROOT_DIR="${MCI_ROOT}" -DNFM_ROOT_DIR="${NFM_ROOT}" -DVMC_ROOT_DIR="${VMC_ROOT}" -DFFNN_ROOT_DIR="${FFNN_ROOT}" -DGSL_ROOT_DIR="${GSL_ROOT}" ..

if [ "$1" = "" ]; then
  make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null)
else
  make -j$1
fi
