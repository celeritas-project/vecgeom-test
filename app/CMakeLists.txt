#---------------------------------*-CMake-*----------------------------------#
# Copyright 2020 UT-Battelle, LLC and other VecGeomTest Developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#----------------------------------------------------------------------------#

add_executable(vecgeom-test vecgeom-test.cc vgthelper.cu)
target_link_libraries(vecgeom-test VecGeom::VecGeom celeritas)
set_target_properties(vecgeom-test PROPERTIES
  CXX_STANDARD 14
  CXX_EXTENSIONS OFF
)

install(TARGETS vecgeom-test
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

#---------------------------------------------------------------------------##
