#---------------------------------*-CMake-*----------------------------------#
# Copyright 2020 UT-Battelle, LLC and other VecGeomTest Developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#----------------------------------------------------------------------------#

cmake_minimum_required(VERSION 3.10)
project(VecGeomTest VERSION 0.0.1 LANGUAGES CXX)

include(GNUInstallDirs)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

#---------------------------------------------------------------------------##
# OPTIONS
#---------------------------------------------------------------------------##

option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(VGT_USE_CUDA "Build with CUDA support" OFF)

# Set up rpaths
if (NOT DEFINED CMAKE_INSTALL_RPATH_USE_LINK_PATH)
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON CACHE STRING
      "Inform installed binaries of external library rpaths")
endif()

if (BUILD_SHARED_LIBS)
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_FULL_LIBDIR}" CACHE STRING
      "Inform installed binaries of internal library rpaths")
endif()

#---------------------------------------------------------------------------##
# DEPENDENCIES
#---------------------------------------------------------------------------##

find_package(VecGeom REQUIRED)

if (VGT_USE_CUDA)
  message(FATAL_ERROR "CUDA support isn't enabled yet")
  enable_language(CUDA)
endif()

#---------------------------------------------------------------------------##
# DEMO APP
#---------------------------------------------------------------------------##

add_executable(vecgeom-test app/vecgeom-test.cc)
target_link_libraries(vecgeom-test VecGeom::VecGeom)
set_target_properties(vecgeom-test PROPERTIES
  CXX_STANDARD 14
  CXX_EXTENSIONS OFF
)

install(TARGETS vecgeom-test
  RUNTIME DESTINATION	${CMAKE_INSTALL_BINDIR}
)

#---------------------------------------------------------------------------##
