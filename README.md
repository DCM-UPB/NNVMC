# NNVMC

C++ Library for Variational Monte Carlo simulations using a Feed Forward Neural Network as trial wavefunction.

To get you started, there is a user manual pdf in `doc/` and in `examples/` there are several basic examples.

Most subdirectories come with a `README.md` file, explaining the purpose and what you need to know.



# Supported Systems

Currently, we automatically test the library on Arch Linux (GCC 8) and MacOS (with clang as well as GCC).
However, in principle any system with C++11 supporting compiler should work, at least if you manage to install all dependencies.



# Build the library

Make sure you have all 4 NNVMC base libraries (FFNN, MCI, NFM and VMC++) installed in known paths.
Also, we require the Autotools build system to be available.
Optionally, if you have valgrind installed on your system, it will be used to check for memory errors when running unittests.

If you have the libraries in non-standard paths or want to use custom compiler flags, copy a little script:

   `cp script/config_template.sh config.sh`

Now edit `config.sh` to your needs and before proceeding run:

   `source config.sh`

If you now have prepared your system, you may setup the build environment by using the following script:

   `./autogen.sh`

Now you want to configure the build process for your platform by invoking:

   `./configure`

Finally, you are ready to compile all the code files in our repository together, by:

   `make` or `make -jN`

where N is the number of parallel threads used by make. Alternatively, you may use the following make targets to build only subparts of the project:

   `make lib`, `make test`, `make benchmark`, `make examples`


As long as you changed, but didn't remove or add source files, it is sufficient to only run `make` again to rebuild.

If you however removed old or added new code files under `src/`, you need to first update the source file lists and include links. Do so by invoking from root folder:

   `make update-sources`

NOTE: All the subdirectories of test, benchmark and examples support calling `make` inside them to recompile local changes.



# Installation

To install the freshly built library and headers into the standard system paths, run (usually sudo is required):
  `make install`

If you however want to install the library under a custom path, before installing you have to use
  `./configure --prefix=/your/absolute/path`



# Build options

You may enable special compiler flags by using one or more of the following options after `configure`:

   `--enable-debug` : Enables flags (like \-g and \-O0) suitable for debugging

   `--enable-coverage` : Enables flags to generate test coverage reports via gcov

   `--enable-profiling` : Enables flags to generate performance profiles for benchmarks




## Multi-threading: OpenMP

This library supports multi-threading computation with a shared memory paradigm, thanks to OpenMP.

To activate this feature use `--enable-openmp` at configuration. Currently it is not recommended to use this for most cases.
