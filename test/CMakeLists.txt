#---------------------------------*-CMake-*----------------------------------#
# Copyright 2020 UT-Battelle, LLC and other VecGeomTest Developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#----------------------------------------------------------------------------#

add_executable(test_devptr test-devptr.cc)
target_link_libraries(test_devptr VecGeom::VecGeom celeritas
  ${CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES})
target_link_directories(test_devptr PRIVATE ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
target_include_directories(test_devptr SYSTEM PRIVATE
  ${CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES})
set_target_properties(test_devptr PROPERTIES
  CXX_STANDARD 14
  CXX_EXTENSIONS OFF
)

install(TARGETS test_devptr
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

#---------------------------------------------------------------------------##
