/*
 *  multimeter.h
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





#ifndef MULTIMETER_H
#define MULTIMETER_H
#include <stdio.h>
#include <string>
#include <vector>
#include "base_neuron.h"

class Record
{
 public:
  bool data_vect_flag_;
  bool out_file_flag_;
  std::vector<std::vector<float> > data_vect_;
  std::vector<BaseNeuron*> neuron_vect_;
  std::string file_name_;
  std::vector<std::string> var_name_vect_;
  std::vector<int> i_neuron_vect_;
  std::vector<int> port_vect_;
  std::vector<float*> var_pt_vect_;
  FILE *fp_;

  Record(std::vector<BaseNeuron*> neur_vect, std::string file_name,
	 std::vector<std::string> var_name_vect,
	 std::vector<int> i_neur_vect, std::vector<int> port_vect);

  int OpenFile();
  
  int CloseFile();
  
  int WriteRecord(float t);

};
  
class Multimeter
{
 public:
  std::vector<Record> record_vect_;

  int CreateRecord(std::vector<BaseNeuron*> neur_vect,
		   std::string file_name,
		   std::vector<std::string> var_name_vect,
		   std::vector<int> i_neur_vect,
		   std::vector<int> port_vect);
  int OpenFiles();

  int CloseFiles();

  int WriteRecords(float t);

  std::vector<std::vector<float> > *GetRecordData(int i_record);
	     
};

#endif
