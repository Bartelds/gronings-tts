#!/usr/bin/env bash

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

download_dir=$1

# check arguments
if [ $# != 1 ]; then
    echo "Usage: $0 <download_dir>"
    exit 1
fi

set -euo pipefail

cwd=$(pwd)

# download jets model
if [ ! -e "${download_dir}/imdanboy/jets" ]; then
    mkdir -p "${download_dir}/imdanboy"
    cd "${download_dir}/imdanboy"
    git clone https://huggingface.co/imdanboy/jets
    cd "${cwd}"
    echo "successfully prepared jets model."
else
    echo "jets model already exists. skipped."
fi

# download vits model
if [ ! -e "${download_dir}/espnet/kan-bayashi_ljspeech_vits" ]; then
    mkdir -p "${download_dir}/espnet"
    cd "${download_dir}/espnet"
    git clone https://huggingface.co/espnet/kan-bayashi_ljspeech_vits
    cd "${cwd}"
    echo "successfully prepared vits model."
else
    echo "vits model already exists. skipped."
fi
