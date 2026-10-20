#include "vp_interface.h"

#include "connection_generator.h"
#include "thread_aligned_array.h"

using namespace sapi;

void test_manager() {
  TAArray<ConnectionGenerator> taa;
  {
    NumericFunctor cf;
    cf.df_.func_ = DISPLACEMENT_FUNCTION::DISTANCE;
    NFCollection cfc;
    cfc.weight_functor_ = cf;
    cfc.delay_functor_ = cf;
    cfc.probability_functor_ = cf;

    ConnectionGenerator cg;
    cg.rule_ = CONNECTION_RULE::PAIRWISE_BERNOULLI;
    cg.numeric_functors_ = cfc;
    taa.clone(cg);

#pragma omp parallel default(shared)
    {
      const auto item = taa.get_local_thread_item();
      assert(item->rule_ == CONNECTION_RULE::PAIRWISE_BERNOULLI &&
             item->numeric_functors_.weight_functor_.is_initialized() &&
             item->numeric_functors_.delay_functor_.is_initialized() &&
             item->numeric_functors_.probability_functor_.is_initialized() &&
             item->connection_counts_ == 0 && item->check_parameters());
    }

    cg.rule_ = CONNECTION_RULE::FIXED_IN_DEGREE;
    cg.connection_counts_ = 10;
    cg.numeric_functors_.probability_functor_ = NumericFunctor();
    taa.clone(cg);

#pragma omp parallel default(shared)
    {
      const auto item = taa.get_local_thread_item();
      assert(item->rule_ == CONNECTION_RULE::FIXED_IN_DEGREE &&
             item->numeric_functors_.weight_functor_.is_initialized() &&
             item->numeric_functors_.delay_functor_.is_initialized() &&
             !item->numeric_functors_.probability_functor_.is_initialized() &&
             item->connection_counts_ == 10 && item->check_parameters());
    }
  }

#pragma omp parallel default(shared)
  {
    const auto item = taa.get_local_thread_item();
    assert(item->rule_ == CONNECTION_RULE::FIXED_IN_DEGREE &&
           item->numeric_functors_.weight_functor_.is_initialized() &&
           item->numeric_functors_.delay_functor_.is_initialized() &&
           !item->numeric_functors_.probability_functor_.is_initialized() &&
           item->connection_counts_ == 10 && item->check_parameters());
  }
}

int main() {
  init_omp();
  test_manager();
  return 0;
}
