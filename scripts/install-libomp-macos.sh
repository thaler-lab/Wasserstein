#!/usr/bin/env bash

wget https://github.com/llvm/llvm-project/releases/download/llvmorg-11.0.0/openmp-11.0.0.src.tar.xz
tar xf openmp-11.0.0.src.tar.xz
rm openmp-11.0.0.src.tar.xz
cd openmp-11.0.0.src

mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$(brew --prefix) -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLIBOMP_INSTALL_ALIASES=OFF -Wno-dev ..
make install
cd ../..
rm -rf openmp-11.0.0.src
