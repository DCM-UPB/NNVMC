include test.am

SUBDIRS = ut1

TESTS = exe ut1/exe
AUTOMAKE_OPTIONS = parallel-tests
if !OPENMP
if VALGRIND_ENABLED
  @VALGRIND_CHECK_RULES@
  LOG_COMPILER = $(LIBTOOL) e $(VALGRIND) --leak-check=full --track-origins=yes
else
  LOG_COMPILER = $(LIBTOOL) e
endif
endif
