//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file array.h
//---------------------------------------------------------------------------//
#ifndef base_array_h
#define base_array_h

#include <cstddef>

#include "Macros.h"

namespace celeritas {

template <typename ElementType, std::size_t Extent>
class span;

//---------------------------------------------------------------------------//
/*!
 * Fixed-size simple array for storage.
 *
 * This isn't fully standards-compliant with std::array: there's no support for
 * N=0 for example.
 */
template <typename T, std::size_t N>
struct array {
  //@{
  //! Type aliases
  using value_type = T;
  using size_type = std::size_t;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  using reference = value_type&;
  using const_reference = const value_type&;
  using iterator = pointer;
  using const_iterator = const_pointer;
  //@}

  // >>> DATA

  T data_[N];

  // >>> ACCESSORS

  //@{
  //! Element access
  CELERITAS_HOST_DEVICE const_reference operator[](size_type i) const {
    return data_[i];
  }
  CELERITAS_HOST_DEVICE reference operator[](size_type i) { return data_[i]; }
  CELERITAS_HOST_DEVICE const_reference front() const { return data_[0]; }
  CELERITAS_HOST_DEVICE reference front() { return data_[0]; }
  CELERITAS_HOST_DEVICE const_reference back() const { return data_[N - 1]; }
  CELERITAS_HOST_DEVICE reference back() { return data_[N - 1]; }
  CELERITAS_HOST_DEVICE const_pointer data() const { return data_; }
  CELERITAS_HOST_DEVICE pointer data() { return data_; }
  //@}

  //@{
  //! Iterators
  CELERITAS_HOST_DEVICE iterator begin() { return data_; }
  CELERITAS_HOST_DEVICE iterator end() { return data_ + N; }
  CELERITAS_HOST_DEVICE const_iterator begin() const { return data_; }
  CELERITAS_HOST_DEVICE const_iterator end() const { return data_ + N; }
  CELERITAS_HOST_DEVICE const_iterator cbegin() const { return data_; }
  CELERITAS_HOST_DEVICE const_iterator cend() const { return data_ + N; }
  //@}

  //@{
  //! Capacity
  CELERITAS_HOST_DEVICE bool empty() const { return N == 0; }
  CELERITAS_HOST_DEVICE size_type size() const { return N; }
  //@}

  //@{
  //! Operations
  CELERITAS_HOST_DEVICE void fill(const_reference value) const {
    for (size_type i = 0; i != N; ++i) data_[i] = value;
  }
  //@}
};

//---------------------------------------------------------------------------//

template <typename T, std::size_t N>
inline CELERITAS_HOST_DEVICE bool operator==(const array<T, N>& lhs,
                                             const array<T, N>& rhs);

template <typename T, std::size_t N>
CELERITAS_HOST_DEVICE bool operator!=(const array<T, N>& lhs,
                                      const array<T, N>& rhs);

template <typename T, std::size_t N>
CELERITAS_HOST_DEVICE void axpy(T a, const array<T, N>& x, array<T, N>* y);

//---------------------------------------------------------------------------//
//! Get a mutable fixed-size view to an array
template <typename T, std::size_t N>
constexpr CELERITAS_HOST_DEVICE span<T, N> make_span(array<T, N>& x) {
  return {x.data(), N};
}

//---------------------------------------------------------------------------//
//! Get a constant fixed-size view to an array
template <typename T, std::size_t N>
constexpr CELERITAS_HOST_DEVICE span<const T, N> make_span(
    const array<T, N>& x) {
  return {x.data(), N};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

#endif  // base_array_h
