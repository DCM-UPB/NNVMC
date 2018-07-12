# ===============================================================================
# Adapted from https://www.gnu.org/software/autoconf-archive/ax_code_coverage.html
# ================================================================================
#
# SYNOPSIS
#
#   AX_COVERAGE()
#
# DESCRIPTION
#
#   Defines COVERAGE_CPPFLAGS, COVERAGE_CFLAGS,
#   COVERAGE_CXXFLAGS and COVERAGE_LDFLAGS which should be included
#   in the CPPFLAGS, CFLAGS CXXFLAGS and LIBS/LIBADD variables of every
#   build target (program or library) which should be built with code
#   coverage support. Also add rules using AX_ADD_AM_MACRO_STATIC; and
#   $enable_coverage which can be used in subsequent configure output.
#   COVERAGE_ENABLED is defined and substituted, and corresponds to the
#   value of the --enable-coverage option, which defaults to being
#   disabled.
#
#   Test also for gcov program and create GCOV variable that could be
#   substituted.
#
#   Note that all optimization flags in CFLAGS must be disabled when code
#   coverage is enabled.
#
#   Usage example:
#
#   configure.ac:
#
#     AX_COVERAGE
#
#   Makefile.am:
#
#     include $(top_srcdir)/aminclude_static.am
#
#     my_program_LIBS = ... $(COVERAGE_LDFLAGS) ...
#     my_program_CPPFLAGS = ... $(COVERAGE_CPPFLAGS) ...
#     my_program_CFLAGS = ... $(COVERAGE_CFLAGS) ...
#     my_program_CXXFLAGS = ... $(COVERAGE_CXXFLAGS) ...
#
#     clean-local: coverage-clean
#     dist-clean-local: coverage-dist-clean
#
#   This code was derived from Makefile.decl in GLib, originally licensed
#   under LGPLv2.1+.
#
# LICENSE
#
#   Copyright (c) 2012, 2016 Philip Withnall
#   Copyright (c) 2012 Xan Lopez
#   Copyright (c) 2012 Christian Persch
#   Copyright (c) 2012 Paolo Borelli
#   Copyright (c) 2012 Dan Winship
#   Copyright (c) 2015,2018 Bastien ROUCARIES
#
#   Modified in 2018 by Jan Kessler.
#
#   This library is free software; you can redistribute it and/or modify it
#   under the terms of the GNU Lesser General Public License as published by
#   the Free Software Foundation; either version 2.1 of the License, or (at
#   your option) any later version.
#
#   This library is distributed in the hope that it will be useful, but
#   WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser
#   General Public License for more details.
#
#   You should have received a copy of the GNU Lesser General Public License
#   along with this program. If not, see <https://www.gnu.org/licenses/>.

#serial 31

m4_define(_AX_COVERAGE_RULES,[
AX_ADD_AM_MACRO_STATIC([

# Code coverage
#
if COVERAGE_ENABLED
 ifeq (\$(abs_builddir), \$(abs_top_builddir))

coverage-clean:
	-find . \\( -name \"*.gcda\" -o -name \"*.gcno\" -o -name \"*.gcov\" \\) -delete

coverage-dist-clean:
A][M_DISTCHECK_CONFIGURE_FLAGS = \$(A][M_DISTCHECK_CONFIGURE_FLAGS) --disable-coverage

 else # ifneq (\$(abs_builddir), \$(abs_top_builddir))

check-coverage: check

coverage-clean:

coverage-dist-clean:

 endif # ifeq (\$(abs_builddir), \$(abs_top_builddir))

else #! COVERAGE_ENABLED

# Use recursive makes in order to ignore errors during check
check-coverage:
	@echo \"Need to reconfigure with --enable-coverage\"

coverage-clean:

coverage-dist-clean:

endif #COVERAGE_ENABLED

.PHONY: check-coverage coverage-dist-clean coverage-clean

])
])

AC_DEFUN([_AX_COVERAGE_ENABLED],[
	AX_CHECK_GNU_MAKE([],[AC_MSG_ERROR([not using GNU make that is needed for coverage])])
	AC_REQUIRE([AX_ADD_AM_MACRO_STATIC])
	# check for gcov
	AC_CHECK_TOOL([GCOV],
		  [$_AX_COVERAGE_GCOV_PROG_WITH],
		  [:])
	AS_IF([test "X$GCOV" = "X:"],
	      [AC_MSG_ERROR([gcov is needed to do coverage])])
	AC_SUBST([GCOV])

	dnl Build the code coverage flags
	dnl Define COVERAGE_LDFLAGS for backwards compatibility
	COVERAGE_CPPFLAGS=""
	COVERAGE_CFLAGS="-O0 -g -fprofile-arcs -ftest-coverage"
	COVERAGE_CXXFLAGS="-O0 -g -fprofile-arcs -ftest-coverage"
        COVERAGE_LDFLAGS="--coverage"

	dnl Check if g++ is being used

	AC_SUBST([COVERAGE_CPPFLAGS])
	AC_SUBST([COVERAGE_CFLAGS])
	AC_SUBST([COVERAGE_CXXFLAGS])
	AC_SUBST([COVERAGE_LDFLAGS])
])

AC_DEFUN([AX_COVERAGE],[
	dnl Check for --enable-coverage

	# allow to override gcov location
	AC_ARG_WITH([gcov],
	  [AS_HELP_STRING([--with-gcov[=GCOV]], [use given GCOV for coverage (GCOV=gcov).])],
	  [_AX_COVERAGE_GCOV_PROG_WITH=$with_gcov],
	  [_AX_COVERAGE_GCOV_PROG_WITH=gcov])

	AC_MSG_CHECKING([whether to build with code coverage support])
	AC_ARG_ENABLE([coverage],
	  AS_HELP_STRING([--enable-coverage],
	  [Whether to enable code coverage support]),,
	  enable_coverage=no)

	AM_CONDITIONAL([COVERAGE_ENABLED], [test "x$enable_coverage" = xyes])
	AC_SUBST([COVERAGE_ENABLED], [$enable_coverage])
	AC_MSG_RESULT($enable_coverage)

	AS_IF([ test "x$enable_coverage" = xyes ], [
		_AX_COVERAGE_ENABLED
	      ])

	_AX_COVERAGE_RULES
])

