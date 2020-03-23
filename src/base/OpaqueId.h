//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file OpaqueId.h
//---------------------------------------------------------------------------//
#ifndef base_OpaqueId_h
#define base_OpaqueId_h

#include "Macros.h"

namespace celeritas {
//---------------------------------------------------------------------------//
/*!
 * Type-safe container for an integer identifier.
 * \tparam Instantiator Class that uses the indexing type.
 * \tparam T Value type for the ID.
 *
 * This allows type-safe, read-only indexing/access for a class. The value is
 * 'true' if it's assigned, 'false' if invalid.
 */
template <class Instantiator, class T = std::size_t>
class OpaqueId {
 public:
  //@{
  //! Type aliases
  using value_type = T;
  //@}

 public:
  // Default to invalid state
  OpaqueId() = default;

  //! Construct explicitly with stored value
  CELERITAS_HOST_DEVICE explicit OpaqueId(value_type index) : value_(index) {
    // assert(index != InvalidValue());
  }

  //! Whether this ID is in a valid (assigned) state
  CELERITAS_HOST_DEVICE explicit operator bool() const {
    return value_ != InvalidValue();
  }

  //! Get the ID's value for interfacing to type-unsafe code
  CELERITAS_HOST_DEVICE value_type Get() const {
    // assert(*this);
    return value_;
  }

 private:
  //! Value indicating the ID is not assigned
  static constexpr CELERITAS_HOST_DEVICE value_type InvalidValue() {
    return static_cast<value_type>(-1);
  }

  value_type value_ = InvalidValue();

  template <class I2, class T2>
  friend CELERITAS_HOST_DEVICE bool operator==(OpaqueId<I2, T2>,
                                               OpaqueId<I2, T2>);
};

//---------------------------------------------------------------------------//
//! Test equality
template <class I, class T>
CELERITAS_HOST_DEVICE bool operator==(OpaqueId<I, T> lhs, OpaqueId<I, T> rhs) {
  return lhs.value_ == rhs.value_;
}

//! Test inequality
template <class I, class T>
CELERITAS_HOST_DEVICE bool operator!=(OpaqueId<I, T> lhs, OpaqueId<I, T> rhs) {
  return !(lhs == rhs);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

#endif  // base_OpaqueId_h
