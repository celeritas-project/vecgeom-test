#---------------------------------*-CMake-*----------------------------------#
# Copyright 2020 UT-Battelle, LLC and other VecGeomTest Developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

FindVecGeom
-----------

Find VecGeom and define modern CMake targets.

.. code-block:: cmake

   find_package(VecGeom REQUIRED)
   target_link_libraries(<MYTARGET> VecGeom::VecGeom)

This script changes the scope of VecGeom definitions from *global* to
*target*-based.

#]=======================================================================]

# Save compile definitions to reverse VecGeom's global add_definitions call
get_property(_SAVED_COMPILE_DEFS DIRECTORY PROPERTY COMPILE_DEFINITIONS)

find_package(VecGeom QUIET NO_MODULE)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(VecGeom HANDLE_COMPONENTS CONFIG_MODE)

# Restore global compile definitions
set_property(DIRECTORY PROPERTY COMPILE_DEFINITIONS "${_SAVED_COMPILE_DEFS}")

set(_VG_TARGET "VecGeom::VecGeom")
if(VECGEOM_FOUND AND NOT TARGET "${_VG_TARGET}")
  # Remove leading -D from vecgeom definitions
  foreach(_DEF IN LISTS VECGEOM_DEFINITIONS)
    string(REGEX REPLACE "^-D" "" _DEF "${_DEF}")
    list(APPEND VECGEOM_DEF_LIST "${_DEF}")
  endforeach()

  # Split libraries into "primary" and "dependencies"
  # XXX POP_BACK requires CMake 3.15 or higher
  # set(VECGEOM_DEP_LIBRARIES VECGEOM_LIBRARIES)
  # list(POP_BACK VECGEOM_DEP_LIBRARIES _VG_LIBRARY)
  list(GET VECGEOM_LIBRARIES -1 _VG_LIBRARY)
  set(VECGEOM_DEP_LIBRARIES "${VECGEOM_LIBRARIES}")
  list(REMOVE_AT VECGEOM_DEP_LIBRARIES -1)

  # By default the library path has relative components (../..)
  get_filename_component(VECGEOM_LIBRARY "${_VG_LIBRARY}" REALPATH CACHE)

  set(_VG_INCLUDE_DIRS ${VECGEOM_EXTERNAL_INCLUDES})
  if(NOT VECGEOM_INCLUDE_DIR_NEXT)
    # _NEXT has been removed, or the version is too old
    set(VECGEOM_INCLUDE_DIR_NEXT "${VECGEOM_INCLUDE_DIR}")
    if(NOT IS_DIRECTORY "${VECGEOM_INCLUDE_DIR_NEXT}/VecGeom")
      message(SEND_ERROR "The installed version of VecGeom is too old")
    endif()
  endif()
  # Convert relative path to absolute and add to full list of include dirs
  get_filename_component(_VG_INCLUDE_DIR "${VECGEOM_INCLUDE_DIR_NEXT}" REALPATH CACHE)
  list(APPEND _VG_INCLUDE_DIRS "${_VG_INCLUDE_DIR}")

  add_library("${_VG_TARGET}" IMPORTED UNKNOWN)
  set_target_properties("${_VG_TARGET}" PROPERTIES
    IMPORTED_LOCATION "${VECGEOM_LIBRARY}"
    INTERFACE_LINK_LIBRARIES "${VECGEOM_DEP_LIBRARIES}"
    INTERFACE_INCLUDE_DIRECTORIES "${_VG_INCLUDE_DIRS}"
    INTERFACE_COMPILE_DEFINITIONS "${VECGEOM_DEF_LIST}"
    INTERFACE_COMPILE_OPTIONS
      "$<$<COMPILE_LANGUAGE:CXX>:${VECGEOM_COMPILE_OPTIONS}>"
  )

  if(VECGEOM_CUDA_STATIC_LIBRARY)
    get_filename_component(_VG_CUDA_STATIC "${VECGEOM_CUDA_STATIC_LIBRARY}" REALPATH CACHE)
    set(_VG_CUDA_TARGET "VecGeom::Cuda")
    add_library("${_VG_CUDA_TARGET}" IMPORTED STATIC)
    set_target_properties("${_VG_CUDA_TARGET}" PROPERTIES
      IMPORTED_LOCATION "${_VG_CUDA_STATIC}"
      IMPORTED_LINK_INTERFACE_LANGUAGES CUDA
      CUDA_SEPARABLE_COMPILATION ON
    )
    # Propagate include dependencies as well
    target_link_libraries(${_VG_CUDA_TARGET} INTERFACE ${_VG_TARGET})
  endif()
endif()

#----------------------------------------------------------------------------#
