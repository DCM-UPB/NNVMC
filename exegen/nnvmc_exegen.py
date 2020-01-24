#!/usr/bin/env python

import argparse
import yaml
import subprocess

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
## String containing the CMakeList template ###
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
cml_str = """
cmake_minimum_required(VERSION 3.5)
include(FindPackageHandleStandardArgs)

project(nnvmc_exe LANGUAGES CXX VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
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
find_package_handle_standard_args(libmci DEFAULT_MSG MCI_LIBRARY_DIR MCI_INCLUDE_DIR)

find_path(NFM_INCLUDE_DIR nfm/NoisyFunMin.hpp HINTS "${NFM_ROOT_DIR}/include")
find_library(NFM_LIBRARY_DIR nfm HINTS "${NFM_ROOT_DIR}/build/src/")
find_package_handle_standard_args(libnfm DEFAULT_MSG NFM_LIBRARY_DIR NFM_INCLUDE_DIR)

find_path(VMC_INCLUDE_DIR vmc/VMC.hpp HINTS "${VMC_ROOT_DIR}/include/")
find_library(VMC_LIBRARY_DIR vmc HINTS "${VMC_ROOT_DIR}/build/src/")
find_package_handle_standard_args(libvmc DEFAULT_MSG VMC_LIBRARY_DIR VMC_INCLUDE_DIR)

find_path(QNETS_INCLUDE_DIR qnets/poly/FeedForwardNeuralNetwork.hpp HINTS "${QNETS_ROOT_DIR}/include/")
find_library(QNETS_LIBRARY_DIR qnets HINTS "${QNETS_ROOT_DIR}/build/src/")
find_package_handle_standard_args(libqnets DEFAULT_MSG QNETS_LIBRARY_DIR QNETS_INCLUDE_DIR)

find_path(SANNIFA_INCLUDE_DIR sannifa/Sannifa.hpp HINTS "${SANNIFA_ROOT_DIR}/include/")
find_library(SANNIFA_LIBRARY_DIR sannifa HINTS "${SANNIFA_ROOT_DIR}/build/src/")
find_package_handle_standard_args(libsannifa DEFAULT_MSG SANNIFA_LIBRARY_DIR SANNIFA_INCLUDE_DIR)

find_path(NNVMC_INCLUDE_DIR nnvmc/SimpleNNWF.hpp HINTS "${NNVMC_ROOT_DIR}/include/")
find_library(NNVMC_LIBRARY_DIR nnvmc HINTS "${NNVMC_ROOT_DIR}/build/src/")
find_package_handle_standard_args(libnnvmc DEFAULT_MSG NNVMC_LIBRARY_DIR NNVMC_INCLUDE_DIR)

message(STATUS "Configured CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
message(STATUS "Configured CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

# set header / library paths
include_directories(include/ ${MCI_INCLUDE_DIR} ${NFM_INCLUDE_DIR} ${VMC_INCLUDE_DIR} ${QNETS_INCLUDE_DIR} ${SANNIFA_INCLUDE_DIR} ${NNVMC_INCLUDE_DIR}) # headers

add_executable(nnvmc.exe main.cpp)
target_link_libraries(nnvmc.exe "${MCI_LIBRARY_DIR}" "${NFM_LIBRARY_DIR}" "${VMC_LIBRARY_DIR}" "${QNETS_LIBRARY_DIR}" "${SANNIFA_LIBRARY_DIR}" "${NNVMC_LIBRARY_DIR}" "${GSL_LIBRARIES}" "${MPI_CXX_LIBRARIES}") # shared libs
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
#parser.add_argument('-s', '--store', action='store_true',
#                    help='Store the generated source and CMakeList files.')
args = parser.parse_args()
print(args)


# concatenate all includes, namespaces and
# snippet code into the following strings
include_str = '#include "vmc/MPIVMC.hpp"\n' # we always need that one for Init() / Finalize()
namespace_str = ''
snippet_str = ''

with open(args.setup, 'r') as s_file:
    s = yaml.load(s_file, Loader=yaml.FullLoader)

    print('includes_sys: ' + str(s['includes_sys']))
    print('includes_usr: ' + str(s['includes_usr']))
    print('namespaces:   ' + str(s['namespaces']))
    print('snippets:     ' + str(s['snippets']))
    print()

    # add "global" includes first
    if s['includes_sys'] is not None:
        for incl in s['includes_sys']: # searched in system locations
            include_str += '#include <' + incl + '>\n'
    if s['includes_usr'] is not None:
        for incl in s['includes_usr']:
            include_str += '#include "' + incl + '"\n'

    # add "global" namespaces
    if s['namespaces'] is not None:    
        for nmspc in s['namespaces']:
            namespace_str += "using namespace " + nmspc + ";\n"

    # process the code snippets (and cut out the includes)
    if s['snippets'] is not None:
        for snip in s['snippets']:
            with open(snip, 'r') as snip_file:
                for line in snip_file:
                    if '#include' in line: # add to includes
                        include_str += line
                    else:                  # add to main code
                        snippet_str += line


# Generate source file from the strings
with open('main.cpp', 'w') as o_file:
    # write includes
    o_file.write(include_str)

    # open main function
    o_file.write("int main()\n{\n")

    # write namespaces
    o_file.write(namespace_str)

    # init MPI
    o_file.write("const int myrank = MPIVMC::Init();\n")

    # write main snippet code
    o_file.write(snippet_str)

    # close MPI and main
    o_file.write("MPIVMC::Finalize();\n")
    o_file.write("return 0;\n}\n")


# Generate CMakeLists.txt
with open("CMakeLists.txt", 'w') as cml_file:
    cml_file.write(cml_str)


# Generate build script
with open("build.sh", 'w') as b_file:
    # insert environmental config
    with open(args.config, 'r') as c_file:
        b_file.write(c_file.read())

    b_file.write(build_str)


# Build the executable
print("Building...")
subprocess.call(["sh","build.sh"]) 
