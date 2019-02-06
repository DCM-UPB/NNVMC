#!/bin/sh

MYPATH=$(pwd)
MYLIBS="-lgsl -lmci -lnfm -lvmc -lffnn -lnnvmc -lc10 -lcaffe2 -ltorch -lsannifa"

cd ../../
source script/config.sh
echo ${CXX} ${CXXFLAGS} -I./include ${CPPFLAGS} -o exe ${MYPATH}/main.cpp -L./lib/.libs ${LDFLAGS} ${MYLIBS} -D_GLIBCXX_USE_CXX11_ABI=0
${CXX} ${CXXFLAGS} -I./include ${CPPFLAGS} -o exe ${MYPATH}/main.cpp -L./lib/.libs ${LDFLAGS} ${MYLIBS} -D_GLIBCXX_USE_CXX11_ABI=0
mv exe ${MYPATH}
cd ${MYPATH}
