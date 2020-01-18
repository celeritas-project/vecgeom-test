#---------------------------------*-CMake-*----------------------------------#
# Copyright 2020 UT-Battelle, LLC and other VecGeomTest Developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#----------------------------------------------------------------------------#

add_library(celeritas
  base/ScopeRootMessages.cc
)
target_link_libraries(celeritas PRIVATE ROOT::Core)
target_include_directories(celeritas
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

set_target_properties(celeritas PROPERTIES
  CXX_STANDARD 14
  CXX_EXTENSIONS OFF
)
install(TARGETS celeritas
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

file(GLOB _HEADERS
  "base/*.h"
)
install(FILES ${_HEADERS}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

#---------------------------------------------------------------------------##
