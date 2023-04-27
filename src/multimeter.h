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


/* BeginUserDocs: device, recorder

Short description
+++++++++++++++++

Sampling continuous quantities from neurons

Description
+++++++++++

The ``multimeter`` allows to record analog values from neurons.
Differently from NEST, a multimeter can be created using the command
``CreateRecord`` and takes as input the following parameters:

* a string representing the label of the record
* a list of strings with the name of the parameter to be recorded
* a list of node ids from the nodes to be recorded
* a list of integers indicating the node port to be recorded

Thus, the recording of the membrane potential for a single neuron can be
done as follows:

::

   recorder = nestgpu.CreateRecord('label', ['V_m'], [neuron[0]], [0]})

The lenght of the lists should be the same for all the three list
entries of ``CreateRecord``.

The sampling interval for recordings is the same one as the simulation
resolution (default 0.1 ms) and cannot be changed.

Differently from the NEST multimeter, the recorder should not be connected
with the nodes through a Connect routine since the nodes connected
to the record are specified in the ``CreateRecord`` routine.

The command ``GetRecordData`` returns, after the simulation, an
array containing the values of the parameters recorded for every node
specified in the ``CreateRecord`` routine. In particular the array
has a dimension of ``simulated_time/resolution * n_nodes+1``, where the
first column shows the time simulated and the other columns shows the value of
the parameter recorded for every node. The number of rows and columns
can also be retreived through the commands ``GetRecordDataRows`` and
``GetRecordDataColumns``. Here follows an example:

::

   rows = nestgpu.GetRecordDataRows(recorder)
   columns = nestgpu.GetRecordDataColumns(recorder)
   print("recorder has {} rows and {} columns".format(rows, columns))

   recorded_data = nestgpu.GetRecordData(record)
   
   time = [row[0] for row in recorded_data]
   variable = [row[1] for row in recorded_data]


See also
++++++++

EndUserDocs */



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
