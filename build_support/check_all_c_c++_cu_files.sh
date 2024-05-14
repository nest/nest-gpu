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

# With this script you can easily check all C/C++/CU files contained in
# the src directory of NEST GPU. Internally it uses clang-tidy to do
# the actual check.
#

function make_temp_dir {
  # Create a temporary directory and store its name in a variable.
  TEMPD=$(mktemp -d)

  # Exit if the temp directory wasn't created successfully.
  if [ ! -e "$TEMPD" ]; then
    >&2 echo "Error: failed to create temp directory"
    exit 1    
  fi


  # Make sure the temp directory gets removed on script exit.
  trap "exit 1"           HUP INT PIPE QUIT TERM
  trap 'rm -rf "$TEMPD"'  EXIT
}

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <src-folder-path>"
    exit 1
fi

CMD_DIR=$(dirname $(echo $0))
CLANG_TIDY=${CMD_DIR}/clang-tidy-cuda.sh

if [ ! -f $CLANG_TIDY ]; then
    echo "Error: $CLANG_TIDY file not found in $CMD_DIR folder"
    exit 1
fi

SRC_DIR=$1
if [ -d "$SRC_DIR" ]; then 
    if [ -L "$SRC_DIR" ]; then
	# It is a symlink
	echo "Error: cannot pass a symboloc link as source path"
	exit 1
    fi
else
    echo "Error: source path $SRC_DIR not found"
    exit 1
fi

make_temp_dir
CONF_DIR=${TEMPD}/config
mkdir $CONF_DIR
if [ ! -e "$CONF_DIR" ]; then
    >&2 echo "Error: failed to create $CONF_DIR directory"
    exit 1
fi
CONF_H=${CONF_DIR}/config.h
:>$CONF_H
if [ ! -f $CONF_H ]; then
    echo "Error: cannot create temporary file $CONF_H"
    exit 1
fi


cp $CLANG_TIDY $TEMPD
CLANG_TIDY=$(basename $CLANG_TIDY)
if [ ! -f $TEMPD/$CLANG_TIDY ]; then
    echo "Error: cannot create temporary executable $CLANG_TIDY in folder $TEMPD"
    exit 1
fi

pushd .
cd $SRC_DIR

for fn in $(ls *.cu *.cpp *.cc *.c *.cuh *.hpp *.h); do
    cat $fn | sed 's://<BEGIN-CLANG-TIDY-SKIP>//:#if 0:;s://<END-CLANG-TIDY-SKIP>//:#endif:' > $TEMPD/$fn
    if [ ! -f $TEMPD/$fn ]; then
	echo "Error: cannot create file $TEMPD/$fn"
	popd
	exit 1
    fi
done


cd $TEMPD

PASSED_NUM=0
for fn in $(ls *.cu *.cpp *.cc *.c | grep -v user_m); do
    echo " - Check with $CLANG_TIDY C/C++/CUDA file: $fn"
    #$TEMPD/$CLANG_TIDY --include-path=../../build_cmake/libnestutil/ $fn
    echo "$TEMPD/$CLANG_TIDY --include-path=$CONF_DIR $fn"
    $TEMPD/$CLANG_TIDY --include-path=$CONF_DIR $fn
    if [ $? -eq 0 ]; then
	echo PASSED
	PASSED_NUM=$(($PASSED_NUM + 1))
    else
	popd
	exit 1
    fi

done

popd
echo "Checked $PASSED_NUM files with clang-tidy-cuda.sh"
echo "All tests PASSED"

exit 0
