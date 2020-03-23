//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Macros.h
//---------------------------------------------------------------------------//
#ifndef base_Macros_h
#define base_Macros_h

#ifdef __CUDACC__
#define CELERITAS_HOST_DEVICE __host__ __device__
#else
#define CELERITAS_HOST_DEVICE
#endif

#if !defined(__CUDACC__) || __CUDACC_VER_MAJOR__ >= 10
#include <type_traits>
#define CELERITAS_IS_TRIVIALLY_COPYABLE(T) std::is_trivially_copyable<T>::value
#else
// Older CUDA versions
#define CELERITAS_IS_TRIVIALLY_COPYABLE(T) __has_trivial_copy(T)
#endif

#endif  // base_Macros_h
