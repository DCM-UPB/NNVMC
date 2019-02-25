#!/bin/sh

VALGRIND="valgrind --leak-check=full --track-origins=yes"

cd ../build/test/
for exe in ./ut*.exe; do
    echo
    echo "Running test ${exe}..."
    ${VALGRIND} ${exe}
    echo
done
