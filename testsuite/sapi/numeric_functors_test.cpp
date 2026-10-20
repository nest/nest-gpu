#include "coordinate_geometry.h"
#include "creator_registry.h"
#include "type_erasure_helpers.h"

#include "nf_creators.h"
#include "numeric_functors.h"

using namespace sapi;

void test_uf() {
  const auto func = [](const space_t &val) {
    const auto v = std::exp(val * -0.5);
    return std::fmin((v < 0.5 ? 0. : v), 1.);
  };
  const auto dist_df =
      construct_displacement_functor(DISPLACEMENT_FUNCTION::DISTANCE);
  std::vector<UnaryFunctor> uf_vec;
  uf_vec.reserve(3);
  uf_vec.emplace_back(
      construct_unary_functor(UNARY_FUNCTION::EXPONENTIAL, -0.5));
  uf_vec.emplace_back(
      construct_unary_functor(UNARY_FUNCTION::LOWER_BOUND, 0.5, 0.));
  uf_vec.emplace_back(construct_unary_functor(UNARY_FUNCTION::MIN, 1.));
  for (double exponent = 0; exponent < 16; ++exponent) {
    const auto disp = construct_displacement(
        construct_coord_2D(0, 1. / std::pow(2., exponent)));
    auto val = dist_df(disp);
    for (const auto &uf : uf_vec)
      val = uf(val);
    assert(val == func(disp.get_distance()));
  }
}

conn_param_t manual_cf_value(const Displacement<Coord2D> &dc,
                             const DisplacementFunctor &df,
                             const std::vector<UnaryFunctor> &ufs) {
  auto val = df(dc);
  for (const auto &uf : ufs)
    val = uf(val);
  return std::fmin(std::fmax(val, std::numeric_limits<conn_param_t>::lowest()),
                   std::numeric_limits<conn_param_t>::max());
};

int main() {
  test_uf();

  CreatorRegistry<UnaryFunctor> ufr;
  CreatorRegistry<DisplacementFunctor> dfr;

  initialize_uf_registry(ufr);
  initialize_df_registry(dfr);

  auto uf = ufr.get_creator("identity")->create({});
  assert(uf.func_ == UNARY_FUNCTION::IDENTITY);

  uf = ufr.get_creator("min")->create({10});
  assert(uf.func_ == UNARY_FUNCTION::MIN && uf.var_[0] == 10);

  uf = ufr.get_creator("max")->create({1.0});
  assert(uf.func_ == UNARY_FUNCTION::MAX && uf.var_[0] == 1);

  uf = ufr.get_creator("lower_bound")->create({-1.5, -0.5});
  assert(uf.func_ == UNARY_FUNCTION::LOWER_BOUND && uf.var_[0] == -1.5 &&
         uf.var_[1] == -0.5);

  uf = ufr.get_creator("upper_bound")->create({1.5, 5.0});
  assert(uf.func_ == UNARY_FUNCTION::UPPER_BOUND && uf.var_[0] == 1.5 &&
         uf.var_[1] == 5);

  uf = ufr.get_creator("inverse")->create({});
  assert(uf.func_ == UNARY_FUNCTION::INVERSE);

  uf = ufr.get_creator("factor")->create({3.});
  assert(uf.func_ == UNARY_FUNCTION::FACTOR && uf.var_[0] == 3);

  uf = ufr.get_creator("offset")->create({11.});
  assert(uf.func_ == UNARY_FUNCTION::OFFSET && uf.var_[0] == 11.);

  uf = ufr.get_creator("exponential")->create({0.5});
  assert(uf.func_ == UNARY_FUNCTION::EXPONENTIAL && uf.var_[0] == -1. / 0.5);

  uf = ufr.get_creator("gaussian")->create({5.0, 0.1});
  assert(uf.func_ == UNARY_FUNCTION::GAUSSIAN && uf.var_[0] == -5. &&
         uf.var_[1] == -1 / (2 * 0.1 * 0.1));

  auto df = dfr.get_creator("constant")->create({1.});
  assert(df.func_ == DISPLACEMENT_FUNCTION::CONSTANT && df.var_ == 1.);

  df = dfr.get_creator("distance")->create({});
  assert(df.func_ == DISPLACEMENT_FUNCTION::DISTANCE);

  df = dfr.get_creator("displacement_x")->create({});
  assert(df.func_ == DISPLACEMENT_FUNCTION::DISPLACEMENT_X);

  df = dfr.get_creator("displacement_y")->create({});
  assert(df.func_ == DISPLACEMENT_FUNCTION::DISPLACEMENT_Y);

  df = dfr.get_creator("displacement_z")->create({});
  assert(df.func_ == DISPLACEMENT_FUNCTION::DISPLACEMENT_Z);

  df = dfr.get_creator("distance_x")->create({});
  assert(df.func_ == DISPLACEMENT_FUNCTION::DISTANCE_X);

  df = dfr.get_creator("distance_y")->create({});
  assert(df.func_ == DISPLACEMENT_FUNCTION::DISTANCE_Y);

  df = dfr.get_creator("distance_z")->create({});
  assert(df.func_ == DISPLACEMENT_FUNCTION::DISTANCE_Z);

  const auto disp =
      construct_displacement(construct_coord_2D(1. / 1024., 1. / 512.));
  const DisplacementFunctor dist_df =
      construct_displacement_functor(DISPLACEMENT_FUNCTION::DISTANCE);
  const DisplacementFunctor const_df =
      construct_displacement_functor(DISPLACEMENT_FUNCTION::CONSTANT, 1.);
  std::vector<UnaryFunctor> uf_vec0;
  uf_vec0.reserve(3);
  uf_vec0.emplace_back(
      construct_unary_functor(UNARY_FUNCTION::EXPONENTIAL, -0.5));
  uf_vec0.emplace_back(
      construct_unary_functor(UNARY_FUNCTION::LOWER_BOUND, 0.5, 0.));
  uf_vec0.emplace_back(construct_unary_functor(UNARY_FUNCTION::MIN, 1.));
  std::vector<UnaryFunctor> uf_vec1;
  uf_vec1.reserve(2);
  uf_vec1.emplace_back(
      construct_unary_functor(UNARY_FUNCTION::MAX, 1. / 1024.));
  uf_vec1.emplace_back(construct_unary_functor(UNARY_FUNCTION::INVERSE));
  std::vector<UnaryFunctor> uf_vec2;

  NumericFunctor cf;
  assert(!cf.is_initialized());

  cf.df_ = dist_df;
  cf.ufs_ = uf_vec0;

  assert(cf(disp) == manual_cf_value(disp, dist_df, uf_vec0));

  cf.df_ = dist_df;
  cf.ufs_ = uf_vec1;

  assert(cf(disp) == manual_cf_value(disp, dist_df, uf_vec1));

  cf.df_ = dist_df;
  cf.ufs_ = uf_vec2;

  assert(cf(disp) == manual_cf_value(disp, dist_df, uf_vec2));

  cf.df_ = const_df;
  cf.ufs_ = uf_vec0;

  assert(cf(disp) == manual_cf_value(disp, const_df, uf_vec0));

  cf.df_ = const_df;
  cf.ufs_ = uf_vec1;

  assert(cf(disp) == manual_cf_value(disp, const_df, uf_vec1));

  cf.df_ = const_df;
  cf.ufs_ = uf_vec1;

  assert(cf(disp) == manual_cf_value(disp, const_df, uf_vec2));

  cf = construct_numeric_functor("distance", {}, dfr, {"max", "inverse"},
                                 {{1. / 1024}, {}}, ufr);

  assert(cf(disp) == manual_cf_value(disp, dist_df, uf_vec1));

  return 0;
}
