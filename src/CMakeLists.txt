#---------------------------------*-CMake-*----------------------------------#
# Copyright 2020 UT-Battelle, LLC and other VecGeomTest Developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#----------------------------------------------------------------------------#

#----------------------------------------------------------------------------#
# LIBRARY
#----------------------------------------------------------------------------#

add_library(celeritas_cuda STATIC
  base/KernelParamCalculator.cu
  geom/VGGeometryAdapter.cu
  geom/VGStateContainer.cu
  transport/vgthelper.cu
)
target_link_libraries(celeritas_cuda
  PRIVATE VecGeom::Cuda VecGeom::VecGeom)
set_target_properties(celeritas_cuda PROPERTIES
  CUDA_STANDARD 14
  CUDA_EXTENSIONS OFF
  LINKER_LANGUAGE CUDA
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON # Don't defer symbol resolution
)

add_library(celeritas
  base/ScopeRootMessages.cc
  geom/RootModel.cc
  geom/VGGeometryHost.cc
)
target_link_libraries(celeritas
  PUBLIC VecGeom::VecGeom ROOT::Geom ROOT::Core
  PRIVATE celeritas_cuda)
set_target_properties(celeritas PROPERTIES
  CXX_STANDARD 14
  CXX_EXTENSIONS OFF
)

foreach(_TARGET celeritas celeritas_cuda)
  target_include_directories(${_TARGET}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_BUILD_DIR}>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

  install(TARGETS ${_TARGET}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endforeach()

foreach(_SUBDIR base geom transport)
  file(GLOB _HEADERS
    "${_SUBDIR}/*.h"
  )
  install(FILES ${_HEADERS}
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${_SUBDIR}/"
  )
endforeach()

#---------------------------------------------------------------------------##
