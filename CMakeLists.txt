#---------------------------------*-CMake-*----------------------------------#
# Copyright 2020 UT-Battelle, LLC and other VecGeomTest Developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#----------------------------------------------------------------------------#

cmake_minimum_required(VERSION 3.10)
project(VecGeomTest VERSION 0.0.1 LANGUAGES CXX)
cmake_policy(SET CMP0074 NEW) # Use $ENV{<package>_ROOT} to find packages

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

find_package(ROOT REQUIRED)
find_package(VecGeom REQUIRED)

if (VGT_USE_CUDA)
  if (VECGEOM_CUDA_ARCH STREQUAL "none")
    message(FATAL_ERROR "The VecGeom installation at ${VECGEOM_INSTALL_DIR} "
      "was configured without CUDA, so VGT_USE_CUDA cannot be used")
  endif()
  enable_language(CUDA)
endif()

#---------------------------------------------------------------------------##
# LIBRARY
#---------------------------------------------------------------------------##

add_subdirectory(src)

#---------------------------------------------------------------------------##
# DEMO APP
#---------------------------------------------------------------------------##

add_subdirectory(app)

#---------------------------------------------------------------------------##
