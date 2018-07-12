# Benchmarks

This directory contains benchmarks to test the performance of certain parts of the library.
The `common` subfolder contains common source code and script files that are used by the individual benchmarks in `bench_*` folders.

A conda environment that can be used to run the python script is available as `conda-env.yml`

Currently there are the following benchmarks:

   `bench_xyz`: Placeholder


# Using the benchmarks

Enter the desired benchmark's directory and execute:
   `make run-benchmark`

Instead you may also run all benchmarks together by calling from root or from top benchmark folder:
   `make run-benchmarks`

Each benchmark will write the result into a file `benchmark_new.out`. For visualization execute the plot script:
   `python plot.py benchmark_new.out`

To let the plot compare the new result versus an older one, you have to provide the old output file like:
   `python plot.py benchmark_old.out benchmark_new.out`.

You may also change new/old to more meaningful labels, anything like benchmark_*.out is allowed (except extra _ or . characters).


# Profiling

If you want to use the benchmarks for profiling, recompile the library and benchmarks after configuring
   `./configure --enable-profiling`

Then execute a benchmark via make (!) and afterwards view the profile with:
   `pprof --text exe exe.prof`
