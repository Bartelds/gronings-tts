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
if [ ! -e "${download_dir}/datasets/gos/GroTTS" ]; then
    mkdir -p "${download_dir}/datasets/gos"
    cd "${download_dir}/datasets/gos"
    git clone https://huggingface.co/datasets/gos/GroTTS # placeholder
    cd GroTTS/flacs
    for f in *.tar.gz; do
        tar xzf "$f" "${f%.tar.gz}"
    done
    cd "${cwd}"
    echo "successfully prepared data."
else
    echo "dataset already exists. skipped."
fi
