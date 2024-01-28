#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 input-file"
    return
fi

if [ ! -f .clang-format ]; then
    echo "Error: .clang-format file not found in current directory"
    return
fi

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

cat $1 | sed 's/<<</$$</g;s/>>>/$ >/g;' > tmp1~
clang-format -style=file:.clang-format tmp1~ > tmp2~
cat tmp2~ | sed 's/$$</<<</g;s/$ >/>>>/g;s/$>/>>>/g;' > $1
rm -f tmp1~
rm -f tmp2~
