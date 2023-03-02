#!/bin/sh

# git submodule update --init
# sudo apt-get install libopenblas-dev
# sudo apt-get install python3-pybind11

cmake_path=build
if [ -d "${cmake_path}" ]
then
    sudo rm -rf ${cmake_path}
fi
mkdir ${cmake_path}
cd ${cmake_path}
cmake ../src
make
