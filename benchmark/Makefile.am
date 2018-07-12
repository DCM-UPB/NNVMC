SUBDIRS = bench_xyz

run-benchmarks:
	for dir in $(SUBDIRS); do \
	  if [[ "$$dir" == *"bench_"* ]]; then $(MAKE) -C $$dir run-benchmark; fi; \
	done
