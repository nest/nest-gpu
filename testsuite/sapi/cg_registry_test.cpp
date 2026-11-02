#include "cg_creators.h"
#include "connection_generator.h"
#include "creator_registry.h"
#include "type_erasure_helpers.h"

using namespace sapi;

int main() {
  CreatorRegistry<ConnectionGenerator> cgr;

  initialize_cg_registry(cgr);

  auto cg = cgr.get_creator("pairwise_bernoulli")->create();
  assert(cg.rule_ == CONNECTION_RULE::PAIRWISE_BERNOULLI);

  cg = cgr.get_creator("pairwise_poisson")->create();
  assert(cg.rule_ == CONNECTION_RULE::PAIRWISE_POISSON);

  cg = cgr.get_creator("fixed_indegree")->create();
  assert(cg.rule_ == CONNECTION_RULE::FIXED_IN_DEGREE);

  cg = cgr.get_creator("fixed_outdegree")->create();
  assert(cg.rule_ == CONNECTION_RULE::FIXED_OUT_DEGREE);

  return 0;
}
