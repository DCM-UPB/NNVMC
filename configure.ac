#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT([NNVMC], [0.1], [ithanil@mail.uni-paderborn.de])
: ${CXXFLAGS=""} # delete default optimization user flags
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIRS([m4])
AC_LANG([C++])

AM_INIT_AUTOMAKE([-Wall -Werror foreign subdir-objects])


# --- Options

# Add debug support
AX_CHECK_ENABLE_DEBUG()
AM_CONDITIONAL(DEBUG, test x"$ax_enable_debug" = x"yes")
AM_COND_IF(DEBUG,
    AC_DEFINE(DEBUG, 1, [Define to 0 if this is a release build]),
    AC_DEFINE(DEBUG, 0, [Define to 1 or higher if this is a debug build]))


# Add coverage support (gcov)
AC_PROG_CXX
AX_COVERAGE()
AM_CONDITIONAL(COVERAGE, test "x$enable_coverage" = x"yes")
AM_COND_IF(COVERAGE,
    AC_DEFINE(COVERAGE, 1, [Define to 0 if you don't need code coverage statistics.]),
    AC_DEFINE(COVERAGE, 0, [Define to 1 or higher if you want to generate code coverage statistics.]))


# Add profiling support (gperftools)
AC_ARG_ENABLE(profiling,
  AS_HELP_STRING(
    [--enable-profiling],
    [enable profiling, default: no]),
    [case "${enableval}" in
      yes) profiling=true ;;
      no)  profiling=false ;;
    esac],
    [profiling=false])
AM_CONDITIONAL(PROFILING, test x"$profiling" = x"true")
AM_COND_IF(PROFILING,
    AC_DEFINE(PROFILING, 1, [Define to 0 if you don't need performance profiling.]),
    AC_DEFINE(PROFILING, 0, [Define to 1 or higher if you want to generate performance profiles.]))


# Add OpenMP support
AC_ARG_ENABLE(openmp,
  AS_HELP_STRING(
    [--enable-openmp],
    [enable openmp, default: no]),
    [case "${enableval}" in
      yes) openmp=true ;;
      no)  openmp=false ;;
    esac],
    [openmp=false])
AM_CONDITIONAL(OPENMP, test x"$openmp" = x"true")
AM_COND_IF(OPENMP,
    AC_DEFINE(OPENMP, 1, [Define to 0 if you don't want to use threading via OpenMP.]),
    AC_DEFINE(OPENMP, 0, [Define to 1 or higher if you want to enable OpenMP threading.]))


# --- Checks

AM_PROG_AR

LT_INIT # use libtool

# Checks for programs.
AC_PROG_AWK
AC_PROG_INSTALL
AC_PROG_MAKE_SET
AC_PROG_CC
AC_PROG_CPP
AC_PROG_CXX
AC_PROG_CXXCPP
AC_PROG_LN_S
AC_PROG_RANLIB

# check for c++11 support (and add compiler flag)
AX_CXX_COMPILE_STDCXX(11, noext, mandatory)

# check for valgrind
AX_VALGRIND_CHECK


# Checks for libraries.

# Checks for header files.

# Checks for typedefs, structures, and compiler characteristics.
AC_CHECK_HEADER_STDBOOL
AC_C_INLINE
AC_TYPE_SIZE_T

# Checks for library functions.
AC_CHECK_FUNCS([pow sqrt])


# --- Compiler flags

AC_SUBST([DEBUGFLAGS], ["-g -O0"])
AM_COND_IF(PROFILING, AC_SUBST([OPTFLAGS], ["-O3"]), AC_SUBST([OPTFLAGS], ["-O3 -flto"]))
AC_SUBST([PROF_CFLAGS], ["-g -DWITHGPERFTOOLS"])
AC_SUBST([PROF_LIBS], ["-lprofiler"])
AC_SUBST([OMP_CFLAGS], ["-DOPENMP -fopenmp"])
AC_SUBST([OMP_LIBS], ["-lomp"])

AC_SUBST([AM_CXXFLAGS], ["-Wall"])
AC_SUBST([AM_CPPFLAGS], ["-I$(pwd)/include"])
AC_SUBST([AM_LDFLAGS], ["-L$(pwd)/lib"])
AC_SUBST([COM_LIBS], ["-lmci -lnfm -lvmc -lffnn"])


# --- Makefiles

# output of ./script/generate_ac_config_files.sh
AC_CONFIG_FILES([
lib/Makefile
include/Makefile
test/ut1/Makefile
test/Makefile
examples/ex1/Makefile
examples/ex6/Makefile
examples/ex4/Makefile
examples/ex5/Makefile
examples/test_2p_trap_softcore/sropt_final/Makefile
examples/test_2p_trap_softcore/Makefile
examples/test_2p_trap_softcore/nmsopt_init/Makefile
examples/ex2/Makefile
examples/Makefile
examples/ex3/Makefile
Makefile
benchmark/bench_xyz/Makefile
benchmark/Makefile
])

AC_OUTPUT
