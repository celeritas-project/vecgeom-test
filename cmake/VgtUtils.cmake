#----------------------------------*-CMake-*----------------------------------#
# Copyright 2020 UT-Battelle, LLC and other VecGeomTest Developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

VgtUtils
--------

Functions and macros used to build VecGeomTest.

.. command:: vgt_link_cuda

  Allow C++ code to include Thrust and use the CUDA API (cudaMalloc etc) but not
  call kernels.

    vgt_link_cuda(<target>
                  <INTERFACE|PUBLIC|PRIVATE>)

  ``target``
    Name of the library/executable target.

  ``scope``
    One of ``INTERFACE``, ``PUBLIC``, or ``PRIVATE``.

#]=======================================================================]

function(vgt_link_cuda TARGET SCOPE)
  target_link_libraries(${TARGET} ${SCOPE} ${CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES})
  target_link_directories(${TARGET} ${SCOPE}
    ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
  target_include_directories(${TARGET} SYSTEM ${SCOPE}
    ${CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES})
endfunction()

#-----------------------------------------------------------------------------#
