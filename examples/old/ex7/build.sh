#!/bin/sh

MYPATH=$(pwd)
MYLIBS="-lgsl -lmci -lnfm -lvmc -lnnvmc -lffnn -lnnvmc -lc10 -lcaffe2 -ltorch -lsannifa"

cd ../../
source script/config.sh
echo ${CXX} ${CXXFLAGS} -DUSE_MPI=1 -I./include ${CPPFLAGS} -o exe ${MYPATH}/main.cpp -L./lib/.libs -Wl,-rpath=$(pwd)/lib/.libs ${LDFLAGS} ${MYLIBS} -D_GLIBCXX_USE_CXX11_ABI=1
${CXX} ${CXXFLAGS} -DUSE_MPI=1 -I./include ${CPPFLAGS} -o exe ${MYPATH}/main.cpp -L./lib/.libs -Wl,-rpath=$(pwd)/lib/.libs ${LDFLAGS} ${MYLIBS} -D_GLIBCXX_USE_CXX11_ABI=1
mv exe ${MYPATH}
cd ${MYPATH}
