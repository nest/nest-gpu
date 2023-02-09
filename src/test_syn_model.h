/*
 *  test_syn_model.h
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





#ifndef TESTSYNMODEL_H
#define TESTSYNMODEL_H

#include "syn_model.h"

class TestSynModel : public SynModel
{
 public:
  TestSynModel() {Init();}
  int Init();
};

namespace test_syn_model_ns
{
  enum ParamIndexes {
    i_fact = 0, i_offset,
    N_PARAM
  };

  const std::string test_syn_model_param_name[N_PARAM] = {
    "fact", "offset"
  };

}

#endif
