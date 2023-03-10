#!/bin/sh

# git submodule update --init
# opencv depencies
# sudo apt-get install libgtk2.0-dev pkg-config
# sudo apt-get install build-essential libavcodec-dev libavformat-dev libjpeg.dev libtiff4.dev libswscale-dev

# pip3 install -r requirements.txt 

# sudo apt-get install libopencv-dev
# sudo apt-get install libopenblas-dev
# sudo apt install python3-dev
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
