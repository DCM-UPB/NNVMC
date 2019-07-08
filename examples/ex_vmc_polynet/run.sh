#!/bin/sh
cp plot.py ../../build/examples/
cd ../../build/examples
./ex_vmc_polynet.exe
python plot.py
