#!/bin/sh
cp plot.py ../../build/examples/
cd ../../build/examples
./ex_adamopt.exe
python plot.py
