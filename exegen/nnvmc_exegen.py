#!/usr/bin/env python

import argparse
import yaml

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
## String containing the CMakeList template ###
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
cml_str = """
cmake_minimum_required(VERSION 3.5)
include(FindPackageHandleStandardArgs)

project(nnvmc_exe LANGUAGES CXX VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${USER_CXX_FLAGS}")

# find packages
message(STATUS "Configured MCI_ROOT_DIR: ${MCI_ROOT_DIR}")
message(STATUS "Configured NFM_ROOT_DIR: ${NFM_ROOT_DIR}")
message(STATUS "Configured VMC_ROOT_DIR: ${VMC_ROOT_DIR}")
message(STATUS "Configured QNETS_ROOT_DIR: ${QNETS_ROOT_DIR}")
message(STATUS "Configured SANNIFA_ROOT_DIR: ${SANNIFA_ROOT_DIR}")
message(STATUS "Configured NNVMC_ROOT_DIR: ${NNVMC_ROOT_DIR}")
message(STATUS "Configured GSL_ROOT_DIR: ${GSL_ROOT_DIR}")

if (USE_MPI)
    find_package(MPI)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_MPI=1")
    message(STATUS "MPI_LIBRARIES: ${MPI_LIBRARIES}")
endif ()

find_package(GSL)
message(STATUS "GSL_LIBRARIES: ${GSL_LIBRARIES}")


find_path(MCI_INCLUDE_DIR mci/MCIntegrator.hpp HINTS "${MCI_ROOT_DIR}/include/")
find_library(MCI_LIBRARY_DIR mci HINTS "${MCI_ROOT_DIR}/build/src/")
find_library(MCI_STATIC_LIBRARY_DIR mci_static HINTS "${MCI_ROOT_DIR}/build/src/")
find_package_handle_standard_args(libmci DEFAULT_MSG MCI_LIBRARY_DIR MCI_INCLUDE_DIR)
find_package_handle_standard_args(libmci_static DEFAULT_MSG MCI_STATIC_LIBRARY_DIR MCI_INCLUDE_DIR)

find_path(NFM_INCLUDE_DIR nfm/NoisyFunMin.hpp HINTS "${NFM_ROOT_DIR}/include")
find_library(NFM_LIBRARY_DIR nfm HINTS "${NFM_ROOT_DIR}/build/src/")
find_library(NFM_STATIC_LIBRARY_DIR nfm_static HINTS "${NFM_ROOT_DIR}/build/src/")
find_package_handle_standard_args(libnfm DEFAULT_MSG NFM_LIBRARY_DIR NFM_INCLUDE_DIR)
find_package_handle_standard_args(libnfm_static DEFAULT_MSG NFM_STATIC_LIBRARY_DIR NFM_INCLUDE_DIR)

find_path(VMC_INCLUDE_DIR vmc/VMC.hpp HINTS "${VMC_ROOT_DIR}/include/")
find_library(VMC_LIBRARY_DIR vmc HINTS "${VMC_ROOT_DIR}/build/src/")
find_library(VMC_STATIC_LIBRARY_DIR vmc_static HINTS "${VMC_ROOT_DIR}/build/src/")
find_package_handle_standard_args(libvmc DEFAULT_MSG VMC_LIBRARY_DIR VMC_INCLUDE_DIR)
find_package_handle_standard_args(libvmc_static DEFAULT_MSG VMC_STATIC_LIBRARY_DIR VMC_INCLUDE_DIR)

find_path(QNETS_INCLUDE_DIR qnets/poly/FeedForwardNeuralNetwork.hpp HINTS "${QNETS_ROOT_DIR}/include/")
find_library(QNETS_LIBRARY_DIR qnets HINTS "${QNETS_ROOT_DIR}/build/src/")
find_library(QNETS_STATIC_LIBRARY_DIR qnets_static HINTS "${QNETS_ROOT_DIR}/build/src/")
find_package_handle_standard_args(libqnets DEFAULT_MSG QNETS_LIBRARY_DIR QNETS_INCLUDE_DIR)
find_package_handle_standard_args(libqnets_static DEFAULT_MSG QNETS_STATIC_LIBRARY_DIR QNETS_INCLUDE_DIR)

find_path(SANNIFA_INCLUDE_DIR sannifa/Sannifa.hpp HINTS "${SANNIFA_ROOT_DIR}/include/")
find_library(SANNIFA_LIBRARY_DIR sannifa HINTS "${SANNIFA_ROOT_DIR}/build/src/")
#find_library(SANNIFA_STATIC_LIBRARY_DIR sannifa_static HINTS "${SANNIFA_ROOT_DIR}/build/src/")
find_package_handle_standard_args(libsannifa DEFAULT_MSG SANNIFA_LIBRARY_DIR SANNIFA_INCLUDE_DIR)
#find_package_handle_standard_args(libsannifa_static DEFAULT_MSG SANNIFA_STATIC_LIBRARY_DIR SANNIFA_INCLUDE_DIR)

find_path(NNVMC_INCLUDE_DIR nnvmc/SimpleNNWF.hpp HINTS "${NNVMC_ROOT_DIR}/include/")
find_library(NNVMC_LIBRARY_DIR nnvmc HINTS "${NNVMC_ROOT_DIR}/build/src/")
#find_library(NNVMC_STATIC_LIBRARY_DIR nnvmc_static HINTS "${NNVMC_ROOT_DIR}/build/src/")
find_package_handle_standard_args(libnnvmc DEFAULT_MSG NNVMC_LIBRARY_DIR NNVMC_INCLUDE_DIR)
#find_package_handle_standard_args(libnnvmc_static DEFAULT_MSG NNVMC_STATIC_LIBRARY_DIR NNVMC_INCLUDE_DIR)


message(STATUS "Configured CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
message(STATUS "Configured CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

# set header / library paths
include_directories(include/ ${MCI_INCLUDE_DIR} ${NFM_INCLUDE_DIR} ${VMC_INCLUDE_DIR} ${QNETS_INCLUDE_DIR} ${SANNIFA_INCLUDE_DIR} ${NNVMC_INCLUDE_DIR}) # headers

link_libraries(nnvmc)
add_executable(nnvmc.exe main.cpp)
"""


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
### String containing the build.sh script ###
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
build_str = """
mkdir -p build && cd build
cmake -DCMAKE_CXX_COMPILER="${CXX_COMPILER}" -DUSER_CXX_FLAGS="${CXX_FLAGS}" -DUSE_MPI="${USE_MPI}" -DMCI_ROOT_DIR="${MCI_ROOT}" -DNFM_ROOT_DIR="${NFM_ROOT}" -DVMC_ROOT_DIR="${VMC_ROOT}" -DQNETS_ROOT_DIR="${QNETS_ROOT}" -DSANNIFA_ROOT_DIR="${SANNIFA_ROOT}" -DNNVMC_ROOT_DIR="${NNVMC_ROOT}" -DGSL_ROOT_DIR="${GSL_ROOT}" ..

if [ "$1" = "" ]; then
  make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null)
else
  make -j$1
fi
"""


### --- Main Script --- ###

# Parse command line arguments
parser = argparse.ArgumentParser(prog='nnvmc_exegen')
parser.add_argument('config',
                    help='Path of the config.sh file, which contains the environmental configuration.')
parser.add_argument('setup', 
                    help='Path of the setup.yaml file, which describes the source file setup.')
parser.add_argument('-s', '--store', action='store_true',
                    help='Store the generated source and CMakeList files.')
args = parser.parse_args()
print(args)
parser.print_help()


# Generate source file
with open(args.setup, 'r') as s_file:
    s = yaml.load(s_file, Loader=yaml.FullLoader)

    print(s['includes'])
    print(s['namespaces'])
    print(s['snippets'])

    with open('main.cpp', 'w') as o_file:
        # add global includes first
        for incl in s['includes']:
            o_file.write("#include <" + incl + ">\n")

        # we always need that one for Init() / Finalize()
        o_file.write('#include "vmc/MPIVMC.hpp"\n')

        # open main function
        o_file.write("int main()\n{\n")

        # use "global" namespaces
        for nmspc in s['namespaces']:
            o_file.write("using namespace " + nmspc + ";\n")

        # init MPI
        o_file.write("const int myrank = MPIVMC::Init();\n")

        # fill in snippets
        for snip in s['snippets']:
            with open(snip, 'r') as snip_file:
                o_file.write(snip_file.read())

        # close MPI and main
        o_file.write("MPIVMC::Finalize();\n")
        o_file.write("return 0;\n}\n")


# Generate CMakeLists.txt
with open("CMakeLists.txt", 'w') as cml_file:
    cml_file.write(cml_str)


# Generate run script
with open("build.sh", 'w') as b_file:
    # insert environmental config
    with open(args.config, 'r') as c_file:
        b_file.write(c_file.read())

    b_file.write(build_str)
