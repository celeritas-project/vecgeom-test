#----------------------------------------------------------------------------#
# \file CMakeLists.txt
#
# Copyright (c) 2020 Oak Ridge National Laboratory, UT-Battelle, LLC.
#----------------------------------------------------------------------------#

cmake_minimum_required(VERSION 3.10)
project(VecGeomTest VERSION 0.0.1 LANGUAGES CXX)

include(GNUInstallDirs)

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

find_package(VecGeom)

if (VGT_USE_CUDA)
  message(FATAL_ERROR "CUDA support isn't enabled yet")
  enable_language(CUDA)
endif()

#-----------------------------------------------------------------------------#
# end of CMakeLists.txt
#-----------------------------------------------------------------------------#
