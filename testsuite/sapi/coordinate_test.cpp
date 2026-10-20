#include "coordinate_geometry.h"

using namespace sapi;

int main() {
  const Coord2D coordA = construct_coord_2D(0, 0);
  const Coord2D coordB = construct_coord_2D(0, 1);
  const Coord2D coordC = construct_coord_2D(1, 0);
  const Coord2D coordD = construct_coord_2D(1, 1);

  Coord2D test;

  test = coordA - coordD;
  assert(test.x_ == -1 && test.y_ == -1);

  test = coordD + coordD;
  assert(test.x_ == 2 && test.y_ == 2);

  test = coordD * 3;
  assert(test.x_ == 3 && test.y_ == 3);

  assert(vector_dot(coordA, coordA) == vector_dot(coordA, coordB));
  assert(vector_dot(coordA, coordB) == vector_dot(coordA, coordC));
  assert(vector_dot(coordA, coordC) == vector_dot(coordA, coordD));
  assert(vector_dot(coordA, coordD) == vector_dot(coordD, coordA));
  assert(vector_dot(coordD, coordA) == vector_dot(coordB, coordC));
  assert(vector_dot(coordB, coordC) == vector_dot(coordC, coordB));
  assert(vector_dot(coordA, coordA) == 0);

  assert(vector_dot(coordB, coordB) == vector_dot(coordC, coordC));
  assert(vector_dot(coordB, coordB) == vector_dot(coordB, coordD));
  assert(vector_dot(coordB, coordB) == vector_dot(coordD, coordB));
  assert(vector_dot(coordC, coordC) == vector_dot(coordC, coordD));
  assert(vector_dot(coordC, coordC) == vector_dot(coordD, coordC));
  assert(vector_dot(coordB, coordB) == 1);

  assert(vector_dot(coordD, coordD) == 2);

  assert(vector_dot(coordA, coordA) == vector_norm2(coordA));
  assert(vector_dot(coordB, coordB) == vector_norm2(coordB));
  assert(vector_dot(coordD, coordD) == vector_norm2(coordD));

  assert(distance2(coordA, coordA) == 0);
  assert(distance2(coordB, coordB) == 0);
  assert(distance2(coordC, coordC) == 0);
  assert(distance2(coordD, coordD) == 0);

  assert(distance2(coordA, coordB) == 1);
  assert(distance2(coordB, coordA) == 1);
  assert(distance2(coordA, coordC) == 1);
  assert(distance2(coordC, coordA) == 1);
  assert(distance2(coordB, coordD) == 1);
  assert(distance2(coordD, coordB) == 1);
  assert(distance2(coordC, coordD) == 1);
  assert(distance2(coordD, coordC) == 1);

  assert(distance2(coordA, coordD) == distance2(coordB, coordC));
  assert(distance2(coordD, coordA) == distance2(coordC, coordB));
  assert(distance2(coordD, coordA) == distance2(coordB, coordC));
  assert(distance2(coordA, coordD) == 2);

  assert(distance(coordA, coordB) == 1);
  assert(distance(coordA, coordD) == space_t(M_SQRT2));

  assert(coordA == construct_coord_2D(0, 0));
  assert(coordB == construct_coord_2D(0, 1));
  assert(coordC == construct_coord_2D(1, 0));
  assert(coordD == construct_coord_2D(1, 1));

  assert(midpoint(coordA, coordA) == construct_coord_2D(0, 0));
  assert(midpoint(coordB, coordB) == construct_coord_2D(0, 1));
  assert(midpoint(coordC, coordC) == construct_coord_2D(1, 0));
  assert(midpoint(coordD, coordD) == construct_coord_2D(1, 1));

  assert(midpoint(coordA, coordB) == construct_coord_2D(0, 0.5));
  assert(midpoint(coordB, coordA) == construct_coord_2D(0, 0.5));
  assert(midpoint(coordA, coordC) == construct_coord_2D(0.5, 0));
  assert(midpoint(coordA, coordC) == construct_coord_2D(0.5, 0));
  assert(midpoint(coordB, coordC) == construct_coord_2D(0.5, 0.5));
  assert(midpoint(coordC, coordB) == construct_coord_2D(0.5, 0.5));
  assert(midpoint(coordA, coordD) == construct_coord_2D(0.5, 0.5));
  assert(midpoint(coordC, coordB) == construct_coord_2D(0.5, 0.5));
  assert(midpoint(coordB, coordD) == construct_coord_2D(0.5, 1));
  assert(midpoint(coordD, coordB) == construct_coord_2D(0.5, 1));
  assert(midpoint(coordC, coordD) == construct_coord_2D(1, 0.5));
  assert(midpoint(coordD, coordC) == construct_coord_2D(1, 0.5));

  const Coord2D coordE = construct_coord_2D(1, 2);
  const Coord2D coordF = construct_coord_2D(2, 1);
  const Coord2D coordG = construct_coord_2D(2, 2);

  const Coord2D vectorFE = coordE - coordF;
  const Coord2D vectorFG = coordG - coordF;
  const Coord2D vectorGE = coordE - coordG;
  const Coord2D vectorGF = coordF - coordG;

  auto assert_true = [](auto boolean) { assert(boolean); };
  assert_true(projection_coeff<Coord2D, false>(vectorFE, vectorFG) == 1);
  assert_true(projection_coeff<Coord2D, false>(vectorGE, vectorGF) == 0);
  assert_true(projection_vector<Coord2D, false>(vectorGE, vectorGF) ==
              (coordF - coordG) * 0);
  assert_true(projection_vector<Coord2D, false>(vectorFE, vectorFG) ==
              coordG - coordF);
  assert_true(projection_coord<Coord2D, false>(coordE, coordG, coordF) ==
              coordG);
  assert_true(projection_coord<Coord2D, false>(coordE, coordF, coordG) ==
              coordG);

  assert(circumscribed_center({coordE, coordF, coordG}) ==
         construct_coord_2D(1.5, 1.5));
  assert(circumscribed_center({coordG, coordF, coordE}) ==
         construct_coord_2D(1.5, 1.5));
  assert(circumscribed_center({coordF, coordE, coordG}) ==
         construct_coord_2D(1.5, 1.5));
  assert(circumscribed_center({coordE, coordG, coordF}) ==
         construct_coord_2D(1.5, 1.5));
  assert(circumscribed_center({coordE, coordD, coordF}) ==
         construct_coord_2D(1.5, 1.5));
  assert(circumscribed_center({coordE, coordD, coordG}) ==
         construct_coord_2D(1.5, 1.5));
  assert(circumscribed_center({coordF, coordD, coordG}) ==
         construct_coord_2D(1.5, 1.5));

  const CircumscribedRadius<Coord2D> c_radius =
      construct_circumscribed_radius<Coord2D>(construct_coord_2D(0, 0), 1);
  const space_t len =
      std::sqrt((SP_RELATIVE_EPSILON + 1.) / (SP_RELATIVE_EPSILON - 1.));
  assert(c_radius.coord_in_radius(construct_coord_2D(0.5, 0.5)).first);
  assert(!c_radius.coord_in_radius(construct_coord_2D(1, 1)).first);
  assert(c_radius.coord_in_radius(construct_coord_2D(1, 1), 0.5).first);
  assert(c_radius.coord_in_radius(construct_coord_2D(0, 1)).first);
  assert(c_radius.coord_in_radius(construct_coord_2D(0, len)).first);
  assert(!c_radius
              .coord_in_radius(construct_coord_2D(
                  0, len + std::numeric_limits<space_t>::epsilon()))
              .first);

  const Coord2D angular_offset = create_angular_offset(10);
  const Coord2D point = construct_coord_2D(0, 1);
  // From origin 0,0 displacement Coord is equal to point Coord
  const Coord2D rotated_point = rotate_displacement(point, angular_offset);
  assert(almost_equal(vector_norm2(point), vector_norm2(rotated_point)));

  const Coord2D ref_point = construct_coord_2D(2, 2);
  const Coord2D disp_vector = point - ref_point;
  const Coord2D rotated_t_point =
      rotate_displacement(disp_vector, angular_offset);
  assert(
      almost_equal(vector_norm2(rotated_t_point), distance2(point, ref_point)));

  return 0;
}
