#include "node_containers.h"
#include "test_helpers.h"

using namespace sapi;

int main() {
  const NodeSequence seq0{0, 11};
  assert(seq0 == join_sequences(seq0, seq0));

  const NodeSequence seq1{11, 1};
  const auto seq01 = join_sequences(seq0, seq1);
  assert(seq01.first == 11 && seq01.second == 0);
  assert(seq01 == join_sequences(seq1, seq0));

  const NodeSequence seq2{12, 3};
  const auto seq21 = join_sequences(seq2, seq1);
  assert(seq21.first == 12 && seq21.second == 0);
  assert(seq21 == join_sequences(seq1, seq2));

  const NodeSequence seq3{11, 10};
  const auto seq13 = join_sequences(seq1, seq3);
  assert(seq13.first == 11 && seq13.second == 1);
  assert(seq13 == join_sequences(seq3, seq1));

  const NodeSequence seq4{15, 2};
  const auto seq34 = join_sequences(seq3, seq4);
  assert(seq34.first == 15 && seq34.second == 2);
  assert(seq34 == join_sequences(seq4, seq3));

  const NodeSequence seq5{9, 4};
  const auto seq35 = join_sequences(seq3, seq5);
  assert(seq35.first == 11 && seq35.second == 2);
  assert(seq35 == join_sequences(seq5, seq3));

  const NodeSequence seq6{19, 10};
  const auto seq36 = join_sequences(seq3, seq6);
  assert(seq36.first == 19 && seq36.second == 2);
  assert(seq36 == join_sequences(seq6, seq3));

  const NodeSequence seq7{0, 30};
  const auto seq37 = join_sequences(seq3, seq7);
  assert(seq37.first == 11 && seq37.second == 10);
  assert(seq37 == join_sequences(seq7, seq3));

  return 0;
}
