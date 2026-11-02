/*
 *  coordinates.h
 *
 *  This file is part of NEST GPU.
 *
 *  Copyright (C) 2021 The NEST Initiative
 *
 *  NEST GPU is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NEST GPU is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NEST GPU.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef COORDINATES_H
#define COORDINATES_H

#include <cstring>
#include <string>

#include "numerics.h"

namespace sapi {
struct Coord2D {
  constexpr static const dim_t D = 2;
  space_t x_ = 0.;
  space_t y_ = 0.;

  Coord2D operator+(const Coord2D &coord) const;

  Coord2D operator+(const space_t scalar) const;

  Coord2D operator-(const Coord2D &coord) const;

  Coord2D operator-(const space_t scalar) const;

  Coord2D operator*(const Coord2D &coord) const;

  Coord2D operator*(const space_t scalar) const;

  Coord2D operator/(const Coord2D &coord) const;

  Coord2D operator/(const space_t scalar) const;

  bool operator==(const Coord2D &coord) const;

  bool is_null() const;

  Coord2D negative() const;

  space_t sum() const;

  space_t product() const;

  std::string to_string() const;

  template <typename ForwardIteratorT>
  void copy_to_vec(ForwardIteratorT &&write_it) const;

  template <typename ForwardIteratorT>
  static Coord2D copy_from_vec(ForwardIteratorT &&read_it);

  template <typename ForwardIteratorT>
  void bit_copy_to_vec(ForwardIteratorT &&write_it) const;

  template <typename ForwardIteratorT>
  static Coord2D bit_copy_from_vec(ForwardIteratorT &&read_it);
};

inline Coord2D construct_coord_2D(const space_t x, const space_t y) {
  Coord2D c;
  c.x_ = x;
  c.y_ = y;
  return c;
}

inline Coord2D Coord2D::operator+(const Coord2D &coord) const {
  return construct_coord_2D(x_ + coord.x_, y_ + coord.y_);
}

inline Coord2D Coord2D::operator+(const space_t scalar) const {
  return construct_coord_2D(x_ + scalar, y_ + scalar);
}

inline Coord2D Coord2D::operator-(const Coord2D &coord) const {
  return construct_coord_2D(x_ - coord.x_, y_ - coord.y_);
}

inline Coord2D Coord2D::operator-(const space_t scalar) const {
  return construct_coord_2D(x_ - scalar, y_ - scalar);
}

inline Coord2D Coord2D::operator*(const Coord2D &coord) const {
  return construct_coord_2D(x_ * coord.x_, y_ * coord.y_);
}

inline Coord2D Coord2D::operator*(const space_t scalar) const {
  return construct_coord_2D(x_ * scalar, y_ * scalar);
}

inline Coord2D Coord2D::operator/(const Coord2D &coord) const {
  return construct_coord_2D(x_ / coord.x_, y_ / coord.y_);
}

inline Coord2D Coord2D::operator/(const space_t scalar) const {
  return construct_coord_2D(x_ / scalar, y_ / scalar);
}

inline bool Coord2D::operator==(const Coord2D &coord) const {
  return almost_equal(x_, coord.x_) && almost_equal(y_, coord.y_);
}

inline bool Coord2D::is_null() const {
  return almost_zero(x_) && almost_zero(y_);
}

inline Coord2D Coord2D::negative() const {
  return construct_coord_2D(-x_, -y_);
}

inline space_t Coord2D::sum() const { return x_ + y_; }

inline space_t Coord2D::product() const { return x_ * y_; }

inline std::string Coord2D::to_string() const {
  return "{ x: " + std::to_string(x_) + " , y: " + std::to_string(y_) + " }";
}

template <typename ForwardIteratorT>
inline void Coord2D::copy_to_vec(ForwardIteratorT &&write_it) const {
  *write_it++ = x_;
  *write_it++ = y_;
}

template <typename ForwardIteratorT>
inline Coord2D Coord2D::copy_from_vec(ForwardIteratorT &&read_it) {
  Coord2D coord;
  coord.x_ = *read_it++;
  coord.y_ = *read_it++;
  return coord;
}

template <typename ForwardIteratorT>
inline void Coord2D::bit_copy_to_vec(ForwardIteratorT &&write_it) const {
  std::memcpy(&(*write_it++), &x_, sizeof(space_t));
  std::memcpy(&(*write_it++), &y_, sizeof(space_t));
}

template <typename ForwardIteratorT>
inline Coord2D Coord2D::bit_copy_from_vec(ForwardIteratorT &&read_it) {
  Coord2D coord;
  std::memcpy(&coord.x_, &(*read_it++), sizeof(space_t));
  std::memcpy(&coord.y_, &(*read_it++), sizeof(space_t));
  return coord;
}

struct Coord3D {
  constexpr static const dim_t D = 3;
  space_t x_ = 0.;
  space_t y_ = 0.;
  space_t z_ = 0.;

  Coord3D operator+(const Coord3D &coord) const;

  Coord3D operator+(const space_t scalar) const;

  Coord3D operator-(const Coord3D &coord) const;

  Coord3D operator-(const space_t scalar) const;

  Coord3D operator*(const Coord3D &coord) const;

  Coord3D operator*(const space_t scalar) const;

  Coord3D operator/(const Coord3D &coord) const;

  Coord3D operator/(const space_t scalar) const;

  bool operator==(const Coord3D &coord) const;

  bool is_null() const;

  Coord3D negative() const;

  space_t sum() const;

  space_t product() const;

  std::string to_string() const;

  template <typename ForwardIteratorT>
  void copy_to_vec(ForwardIteratorT &&write_it) const;

  template <typename ForwardIteratorT>
  static Coord3D copy_from_vec(ForwardIteratorT &&read_it);

  template <typename ForwardIteratorT>
  void bit_copy_to_vec(ForwardIteratorT &&write_it) const;

  template <typename ForwardIteratorT>
  static Coord3D bit_copy_from_vec(ForwardIteratorT &&read_it);
};

inline Coord3D construct_coord_3D(const space_t x, const space_t y,
                                  const space_t z) {
  Coord3D c;
  c.x_ = x;
  c.y_ = y;
  c.z_ = z;
  return c;
}

inline Coord3D Coord3D::operator+(const Coord3D &coord) const {
  return construct_coord_3D(x_ + coord.x_, y_ + coord.y_, z_ + coord.z_);
}

inline Coord3D Coord3D::operator+(const space_t scalar) const {
  return construct_coord_3D(x_ + scalar, y_ + scalar, z_ + scalar);
}

inline Coord3D Coord3D::operator-(const Coord3D &coord) const {
  return construct_coord_3D(x_ - coord.x_, y_ - coord.y_, z_ - coord.z_);
}

inline Coord3D Coord3D::operator-(const space_t scalar) const {
  return construct_coord_3D(x_ - scalar, y_ - scalar, z_ - scalar);
}

inline Coord3D Coord3D::operator*(const Coord3D &coord) const {
  return construct_coord_3D(x_ * coord.x_, y_ * coord.y_, z_ * coord.z_);
}

inline Coord3D Coord3D::operator*(const space_t scalar) const {
  return construct_coord_3D(x_ * scalar, y_ * scalar, z_ * scalar);
}

inline Coord3D Coord3D::operator/(const Coord3D &coord) const {
  return construct_coord_3D(x_ / coord.x_, y_ / coord.y_, z_ / coord.z_);
}

inline Coord3D Coord3D::operator/(const space_t scalar) const {
  return construct_coord_3D(x_ / scalar, y_ / scalar, z_ / scalar);
}

inline bool Coord3D::operator==(const Coord3D &coord) const {
  return almost_equal(x_, coord.x_) && almost_equal(y_, coord.y_) &&
         almost_equal(z_, coord.z_);
}

inline bool Coord3D::is_null() const {
  return almost_zero(x_) && almost_zero(y_) && almost_zero(z_);
}

inline Coord3D Coord3D::negative() const {
  return construct_coord_3D(-x_, -y_, -z_);
}

inline space_t Coord3D::sum() const { return x_ + y_ + z_; }

inline space_t Coord3D::product() const { return x_ * y_ * z_; }

inline std::string Coord3D::to_string() const {
  return "{ x: " + std::to_string(x_) + " , y: " + std::to_string(y_) +
         " , z: " + std::to_string(z_) + " }";
}

template <typename ForwardIteratorT>
inline void Coord3D::copy_to_vec(ForwardIteratorT &&write_it) const {
  *write_it++ = x_;
  *write_it++ = y_;
  *write_it++ = z_;
}

template <typename ForwardIteratorT>
inline Coord3D Coord3D::copy_from_vec(ForwardIteratorT &&read_it) {
  Coord3D coord;
  coord.x_ = *read_it++;
  coord.y_ = *read_it++;
  coord.z_ = *read_it++;
  return coord;
}

template <typename ForwardIteratorT>
inline void Coord3D::bit_copy_to_vec(ForwardIteratorT &&write_it) const {
  std::memcpy(&(*write_it++), &x_, sizeof(space_t));
  std::memcpy(&(*write_it++), &y_, sizeof(space_t));
  std::memcpy(&(*write_it++), &z_, sizeof(space_t));
}

template <typename ForwardIteratorT>
inline Coord3D Coord3D::bit_copy_from_vec(ForwardIteratorT &&read_it) {
  Coord3D coord;
  std::memcpy(&coord.x_, &(*read_it++), sizeof(space_t));
  std::memcpy(&coord.y_, &(*read_it++), sizeof(space_t));
  std::memcpy(&coord.z_, &(*read_it++), sizeof(space_t));
  return coord;
}

template <typename CoordT> struct Displacement {
  space_t distance2_ = 0.;
  CoordT displacement_;

  space_t get_distance() const;

  bool operator==(const Displacement &) const;
};

// Forward definition to link with coordinate_geometry.h
template <typename CoordT> space_t vector_norm2(const CoordT &vector);

template <typename CoordT>
inline Displacement<CoordT> construct_displacement(CoordT &&vector) {
  Displacement<CoordT> d;
  d.displacement_ = vector;
  d.distance2_ = vector_norm2(vector);
  return d;
}

template <typename CoordT>
inline space_t Displacement<CoordT>::get_distance() const {
  return std::sqrt(distance2_);
}

template <typename CoordT>
inline bool
Displacement<CoordT>::operator==(const Displacement<CoordT> &d) const {
  return displacement_ == d.displacement_;
}

template <typename CoordT>
using OptDisp = std::pair<bool, Displacement<CoordT>>;

template <typename CoordT> struct CircumscribedRadius {
  space_t radius2_ = 0.;
  CoordT origin_;

  bool disp_in_radius(const Displacement<CoordT> &disp) const;

  bool disp_in_radius(const Displacement<CoordT> &disp, space_t offset) const;

  OptDisp<CoordT> coord_in_radius(const CoordT &coord) const;

  OptDisp<CoordT> coord_in_radius(const CoordT &coord, space_t offset) const;

  OptDisp<CoordT> overlapping_radi(const CircumscribedRadius &) const;

  std::string to_string() const;

  bool operator==(const CircumscribedRadius &) const;
};

template <typename CoordT>
inline CircumscribedRadius<CoordT>
construct_circumscribed_radius(CoordT &&origin, const space_t radius2) {
  CircumscribedRadius<CoordT> cr;
  cr.origin_ = origin;
  cr.radius2_ = radius2;
  return cr;
}

template <typename CoordT>
inline std::string CircumscribedRadius<CoordT>::to_string() const {
  return "{ center: " + origin_.to_string() +
         " , radius: " + std::to_string(std::sqrt(radius2_)) + " }";
}

template <typename CoordT>
inline bool CircumscribedRadius<CoordT>::disp_in_radius(
    const Displacement<CoordT> &disp) const {
  return leq_test(disp.distance2_, radius2_);
}

template <typename CoordT>
inline bool
CircumscribedRadius<CoordT>::disp_in_radius(const Displacement<CoordT> &disp,
                                            space_t offset) const {
  offset = std::sqrt(radius2_) + offset;
  return leq_test(disp.distance2_, offset * offset);
}

template <typename CoordT>
inline OptDisp<CoordT>
CircumscribedRadius<CoordT>::coord_in_radius(const CoordT &coord) const {
  OptDisp<CoordT> od(false, construct_displacement(coord - origin_));
  od.first = disp_in_radius(od.second);
  return od;
}

template <typename CoordT>
inline OptDisp<CoordT>
CircumscribedRadius<CoordT>::coord_in_radius(const CoordT &coord,
                                             space_t offset) const {
  OptDisp<CoordT> od(false, construct_displacement(coord - origin_));
  od.first = disp_in_radius(od.second, offset);
  return od;
}

template <typename CoordT>
inline OptDisp<CoordT> CircumscribedRadius<CoordT>::overlapping_radi(
    const CircumscribedRadius &cr) const {
  return coord_in_radius(cr.origin_, std::sqrt(cr.radius2_));
}

template <typename CoordT>
inline bool
CircumscribedRadius<CoordT>::operator==(const CircumscribedRadius &cr) const {
  return origin_ == cr.origin_ && almost_equal(radius2_, cr.radius2_);
}
} // namespace sapi

#endif
