#!/bin/bash

rm -fr tmpdir~
mkdir tmpdir~

#cp compile_commands.json tmpdir~
cp clang-tidy-cuda.sh tmpdir~

for fn in $(ls *.cu *.cpp *.cc *.c *.cuh *.hpp *.h); do
    cat $fn | sed 's://<BEGIN-CLANG-TIDY-SKIP>//:#if 0:;s://<END-CLANG-TIDY-SKIP>//:#endif:' > tmpdir~/$fn
done

cd tmpdir~

PASSED_NUM=0
for fn in $(ls *.cu *.cpp *.cc *.c | grep -v user_m); do
    echo " - Check with clang-tidy-cuda.sh C/C++/CUDA file: $fn"
    ./clang-tidy-cuda.sh --include-path=../../build_cmake/libnestutil/ $fn
    if [ $? -eq 0 ]; then
	echo PASSED
	PASSED_NUM=$(($PASSED_NUM + 1))
    else
	exit 1
    fi

done

echo "Checked $PASSED_NUM files with clang-tidy-cuda.sh"
exit 0
