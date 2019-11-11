# NNVMC

C++ Library for Variational Monte Carlo simulations using a Feed Forward Neural Network as trial wavefunction.

It is built upon our VMC++ (https://github.com/DCM-UPB/VMCPlusPlus) and QNets (https://github.com/DCM-UPB/QNets) libraries.
Furthermore, the sannifa library (https://github.com/DCM-UPB/sannifa) is required for connection (currently the sannifa build process also requires PyTorch package, which should be optional actually).

In `doc/` there is a user manual in pdf and a config for doxygen.

In `examples/` and `test/` there are examples and tests for the library.


Some subdirectories come with an own `README.md` file which provides further information.


# Supported Systems

Currently, we automatically test the library on Arch Linux (GCC 8) and MacOS (with clang as well as brewed GCC 8).
However, in principle any system with C++11 supporting compiler should work.


# Requirements

- CMake, to use our build process
- master versions of VMC++ (incl. MCI++, NoisyFunMin), QNets and sannifa
- GNU Scientific Library (~2.3+)
- (optional) a MPI implementation, to use parallelized integration
- (optional) valgrind, to run `./run.sh` in `test/`
- (optional) pdflatex, to compile the tex file in `doc/`
- (optional) doxygen, to generate doxygen documentation in `doc/doxygen`


# Build the library

Copy the file `config_template.sh` to `config.sh`, edit it to your liking and then simply execute the command

   `./build.sh`

Note that we build out-of-tree, so the compiled library and executable files can be found in the directories under `./build/`.


# First steps

You may want to read `doc/user_manual.pdf` to get a quick overview of the libraries functionality. However, it is not guaranteed to be perfectly up-to-date and accurate. Therefore, the best way to get your own code started is by studying the examples in `examples/`. See `examples/README.md` for further guidance.


# Multi-threading: MPI

This library supports multi-threaded MC integration with a distributed-memory paradigm, thanks to Message Passing interface (MPI).

To activate this feature, set `USE_MPI=1` inside your config.sh, before building. Please make sure that you always use the same value as you did when compiling the VMC++ library.


# Multi-threading: OpenMP

The used QNets (or PyTorch) libraries also support multi-threaded evaluation with a shared-memory paradigm, thanks to OpenMP. This feature can be enabled and disabled when compiling the QNets or PyTorch libraries.
If you are already using MPI for parallel MC integration, it is usually not beneficial to also use OpenMP for the FFNN.
