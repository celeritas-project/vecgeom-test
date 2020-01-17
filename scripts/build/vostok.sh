#!/bin/sh -e
SOURCE_DIR=$(cd $(dirname $BASH_SOURCE[0]) && git rev-parse --show-toplevel)

printf "\e[2;37mBuilding from ${SOURCE_DIR}\e[0m\n"
cd $SOURCE_DIR
mkdir build 2>/dev/null || true
cd build

export CMAKE_PREFIX_PATH=$(spack location -i vecgeom)
cmake -G Ninja ..
ninja
