#---------------------------------*-CMake-*----------------------------------#
# Copyright 2020 UT-Battelle, LLC and other VecGeomTest Developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

FindVecGeom
--------------

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

set(_VECGEOM_TARGET "VecGeom::VecGeom")
if (VECGEOM_FOUND AND NOT TARGET "${_VECGEOM_TARGET}")
  # Remove leading -D from vecgeom definitions
  foreach(_DEF IN LISTS VECGEOM_DEFINITIONS)
    string(REGEX REPLACE "^-D" "" _DEF "${_DEF}")
    list(APPEND VECGEOM_DEF_LIST "${_DEF}")
  endforeach()

  # Split libraries into "primary" and "dependencies"
  # XXX POP_BACK requires CMake 3.15 or higher
  # set(VECGEOM_DEP_LIBRARIES VECGEOM_LIBRARIES)
  # list(POP_BACK VECGEOM_DEP_LIBRARIES _VECGEOM_LIBRARY)
  list(GET VECGEOM_LIBRARIES -1 _VECGEOM_LIBRARY)
  set(VECGEOM_DEP_LIBRARIES "${VECGEOM_LIBRARIES}")
  list(REMOVE_AT VECGEOM_DEP_LIBRARIES -1)

  # By default the library path has relative components (../..)
  get_filename_component(VECGEOM_LIBRARY "${_VECGEOM_LIBRARY}" REALPATH CACHE)

  set(_NEW_INCLUDE_DIRS ${VECGEOM_EXTERNAL_INCLUDES})
  if (VECGEOM_INCLUDE_DIR_NEXT)
    # Scoped includes: require VecGeom/
    list(APPEND _NEW_INCLUDE_DIRS "${VECGEOM_INCLUDE_DIR_NEXT}")
  else()
    # _NEXT has been removed, or the version is too old
    list(APPEND _NEW_INCLUDE_DIRS "${VECGEOM_INCLUDE_DIR}")
    if (NOT IS_DIRECTORY "${VECGEOM_INCLUDE_DIR}/VecGeom")
      message(SEND_ERROR "The installed version of VecGeom is too old")
    endif()
  endif()

  add_library("${_VECGEOM_TARGET}" IMPORTED UNKNOWN)
  set_target_properties("${_VECGEOM_TARGET}" PROPERTIES
    IMPORTED_LOCATION "${VECGEOM_LIBRARY}"
    INTERFACE_LINK_LIBRARIES "${VECGEOM_DEP_LIBRARIES}"
    INTERFACE_INCLUDE_DIRECTORIES "${_NEW_INCLUDE_DIRS}"
    INTERFACE_COMPILE_DEFINITIONS "${VECGEOM_DEF_LIST}"
  )
endif()

#----------------------------------------------------------------------------#
