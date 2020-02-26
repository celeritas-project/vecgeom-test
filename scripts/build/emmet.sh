#!/bin/sh -e
BUILDSCRIPT_DIR="$(cd "$(dirname $BASH_SOURCE[0])" && pwd)"
SOURCE_DIR="$(cd "${BUILDSCRIPT_DIR}" && git rev-parse --show-toplevel)"

printf "\e[2;37mBuilding from ${SOURCE_DIR}\e[0m\n"
cd $SOURCE_DIR
mkdir build 2>/dev/null || true
cd build

# Note: vecgeom doesn't correctly export RPATH information, so we have to add
# it manually this way
module load vecgeom veccore root

cmake -C ${BUILDSCRIPT_DIR}/emmet.cmake -G Ninja \
  -DCMAKE_INSTALL_PREFIX:PATH=$SOURCE_DIR/install \
  ..
ninja -v
