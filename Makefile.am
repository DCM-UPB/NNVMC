ACLOCAL_AMFLAGS = -I m4

SUBDIRS = lib test benchmark examples

# SUBDIRS targets

lib:
	cd lib && $(MAKE)

test: lib
	cd test && $(MAKE)

benchmark: lib
	cd benchmark && $(MAKE)

examples: lib
	cd examples && $(MAKE)


# Special targets

include-links:
	cd include && $(MAKE) include-links

source-lists:
	cd lib && $(MAKE) source-lists

update-sources: include-links source-lists


run-benchmarks: benchmark
	cd benchmark && $(MAKE) run-benchmarks

.PHONY: lib test benchmark examples source-lists include-links update-sources run-benchmarks
