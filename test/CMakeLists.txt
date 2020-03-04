#---------------------------------*-CMake-*----------------------------------#
# Copyright 2020 UT-Battelle, LLC and other VecGeomTest Developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#----------------------------------------------------------------------------#

# Allow C++ code to include Thrust and use the CUDA API (cudaMalloc etc) but not
# call kernels.
function(vgt_link_cuda TARGET)
  target_link_libraries(${TARGET} ${CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES})
  target_link_directories(${TARGET} PRIVATE
    ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
  target_include_directories(${TARGET} SYSTEM PRIVATE
    ${CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES})
endfunction()

add_executable(test_devptr test-devptr.cc)
target_link_libraries(test_devptr VecGeom::VecGeom celeritas)
set_target_properties(test_devptr PROPERTIES
  CXX_STANDARD 14
  CXX_EXTENSIONS OFF
)
vgt_link_cuda(test_devptr)

install(TARGETS test_devptr
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

#---------------------------------------------------------------------------##
