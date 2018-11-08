#!/bin/bash

echo "Before exporting values"

export CUDA_HOME="/usr/local/cuda"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64"
export PATH=${CUDA_HOME}/bin:${PATH}

echo "After exporting values"