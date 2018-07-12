#!/bin/bash

# Config script for custom library and header paths or C++ compiler choice
#
# If you want to edit this template script/config_template.sh,
# copy it over to somethingl ike config.sh and edit the gitignored copy.
#

# C++ compiler
export CXX="g++"

# C++ flags
export CXXFLAGS=""

# MCIntegrator++ Library
MCI_L="-L$(pwd)/../MCIntegratorPlusPlus/"
MCI_I="-I$(pwd)/../MCIntegratorPlusPlus/src"

# NoisyFunMin Library
NFM_L="-L$(pwd)/../NoisyFunMin/"
NFM_I="-I$(pwd)/../NoisyFunMin/src"

# VMC++ Library
VMC_L="-L$(pwd)/../VMCPlusPlus/"
VMC_I="-I$(pwd)/../VMCPlusPlus/src/"

# FeedForwardNeuralNetwork Library
FFNN_L="-L$(pwd)/../FeedForwardNeuralNetwork/lib/.libs"
FFNN_I="-I$(pwd)/../FeedForwardNeuralNetwork/include/"


# ! DO NOT EDIT THE FOLLOWING !

# linker flags
export LDFLAGS="${MCI_L} ${NFM_L} ${VMC_L} ${FFNN_L}"

# pre-processor flags
export CPPFLAGS="${MCI_I} ${NFM_I} ${VMC_I} ${FFNN_I}"
