#!/bin/bash
# echo a AC_CONFIG_FILES block containing a Makefile for every Makefile.am found
echo "AC_CONFIG_FILES(["
find . -name Makefile.am | sed 's/\.\///g' | sed 's/\.am//g'
echo "])"
