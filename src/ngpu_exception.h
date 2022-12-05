/*
 *  ngpu_exception.h
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

/////////////////////////////////////
// ngpu_exception class definition
// This class handles runtime errors
/////////////////////////////////////

#ifndef NGPUEXCEPTION_H
#define NGPUEXCEPTION_H
#include <string>
#include <cstring>
#include <exception>

///////////////////////////////////
// ngpu_exception class definition
// in case of errors displays a message and stop the execution
//////////////////////////////////
class ngpu_exception: public std::exception
{
  const char *Message; // error message
  
 public:
  // constructors
  ngpu_exception(const char *ch)  {Message=strdup(ch);}
  ngpu_exception(std::string s)  {Message=strdup(s.c_str());}
  // throw method
  virtual const char* what() const throw()
  {
    return Message;
  }
};

#define BEGIN_TRY try
#define END_TRY catch (ngpu_exception &e){ \
    std::cerr << "Error: " << e.what() << "\n"; }			\
  catch (bad_alloc&) { std::cerr << "Error allocating memory." << "\n"; } \
  catch (...) { std::cerr << "Unrecognized error\n"; }


#endif
