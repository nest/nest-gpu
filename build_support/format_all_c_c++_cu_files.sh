#!/bin/bash

#
#   This file is part of NEST GPU.
#
#  Copyright (C) 2021 The NEST Initiative
#
#  NEST GPU is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 2 of the License, or
#  (at your option) any later version.
#
#  NEST GPU is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with NEST GPU.  If not, see <http://www.gnu.org/licenses/>.
#

# With this script you can easily format all C/C++/CU files contained in
# the src directory of NEST GPU. Internally it uses clang-format to do
# the actual formatting.
#
# NEST GPU C/C++/CUDA code should be formatted according to clang-format
# version 17.0.4. If you would like to see how the code will be formatted
# with a different clang-format version, execute e.g.
# `CLANG_FORMAT=clang-format-14 ./format_all_c_c++_cu_files.sh`.
#
# By default the script starts at the current working directory ($PWD), but
# supply a different starting directory as the first argument to the command.

CLANG_FORMAT=${CLANG_FORMAT:-clang-format}
CLANG_FORMAT_FILE=${CLANG_FORMAT_FILE:-${PWD}/.clang-format}

# Drop files that should not be checked
FILES_TO_IGNORE="" # not used now, bult could be used in the future
DIRS_TO_IGNORE="thirdparty" # not used now, bult could be used in the future

function clang_format_cuda {
  if [ ! -f $1 ]; then
    echo "Error: input file $1 not found"
    return
  fi

  if grep -q '$$<' $1; then
    echo 'Error: illegal character sequence in input file: "$$<"'
    return
  fi
  if grep -q '$ >' $1; then
    echo 'Error: illegal character sequence in input file: "$ >"'
    return
  fi
  if grep -q '$>' $1; then
    echo 'Error: illegal character sequence in input file: "$>"'
    return
  fi

  cat $1 | sed 's/<<</$$</g;s/>>>/$ >/g;' > $TEMPD/tmp1~
  echo "CLANG_FORMAT_FILE: $CLANG_FORMAT_FILE"
  clang-format -style=file:$CLANG_FORMAT_FILE $TEMPD/tmp1~ > $TEMPD/tmp2~
  cat $TEMPD/tmp2~ | sed 's/$$</<<</g;s/$ >/>>>/g;s/$>/>>>/g;' > $1
  ls $TEMPD/tmp2~
}  

# Recursively process all C/C++/CUDA files in all sub-directories.
function process_dir {
  dir=$1
  echo "Process directory: $dir"

  if [[ " $DIRS_TO_IGNORE " =~ .*[[:space:]]${dir##*/}[[:space:]].* ]]; then
    echo "   Directory explicitly ignored."
    return
  fi

  for f in $dir/*; do
    if [[ -d $f ]]; then
      # Recursively process sub-directories.
      process_dir $f
    else
      ignore_file=0

      for FILE_TO_IGNORE in $FILES_TO_IGNORE; do
        if [[ $f == *$FILE_TO_IGNORE* ]]; then
          ignore_file=1
          break
        fi
      done

      if [ $ignore_file == 1 ] ; then
        continue
      fi

      case $f in
        *.cpp | *.cc | *.c | *.h | *.hpp | *.cu | *.cuh )
          # Format C/C++/CUDA files.
          echo " - Format C/C++/CUDA file: $f"
          #  $CLANG_FORMAT -i $f
	  clang_format_cuda $f
          ;;
        * )
          # Ignore all other files.
      esac
    fi
  done
}

function help_output {
  echo "The $CLANG_FORMAT_FILE requires clang-format version 13 or later."
  echo "Use like: [CLANG_FORMAT=<clang-format>] ./build_support/`basename $0` [start folder, defaults to '$PWD']"
  exit 0
}

function make_temp_dir {
  # Create a temporary directory and store its name in a variable.
  TEMPD=$(mktemp -d)

  # Exit if the temp directory wasn't created successfully.
  if [ ! -e "$TEMPD" ]; then
    >&2 echo "Failed to create temp directory"
    exit 1    
  fi


  # Make sure the temp directory gets removed on script exit.
  trap "exit 1"           HUP INT PIPE QUIT TERM
  trap 'rm -rf "$TEMPD"'  EXIT
}

make_temp_dir

if [[ ! -f $CLANG_FORMAT_FILE ]]; then
  echo "Cannot find $CLANG_FORMAT_FILE file. Please start '`basename $0`' from the NEST GPU base source directory."
  help_output
fi

if [[ $# -eq 0 ]]; then
  # Start with current directory.
  startdir=$PWD
elif [[ $# -eq 1 ]]; then
  if [[ -d $1 ]]; then
    # Start with given directory.
    startdir=$1
  else
    # Not a directory.
    help_output
  fi
else
  # Two or more arguments...
  help_output
fi

# Start formatting the $startdir and all subdirectories
process_dir $startdir
