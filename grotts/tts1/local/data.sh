#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=-1
stop_stage=2

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    log "stage -1: Data Download"
    local/model_download.sh downloads
    local/data_download.sh downloads
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Data Preparation"
    # set filenames
    scp=data/train/wav.scp
    utt2spk=data/train/utt2spk
    spk2utt=data/train/spk2utt
    text=data/train/text

    # check file existence
    [ ! -e data/train ] && mkdir -p data/train
    [ -e ${scp} ] && rm ${scp}
    [ -e ${utt2spk} ] && rm ${utt2spk}
    [ -e ${spk2utt} ] && rm ${spk2utt}
    [ -e ${text} ] && rm ${text}

    wavs_dir="downloads/datasets/gos/GroTTS/flacs"
    # make scp, utt2spk, and spk2utt
    find "${wavs_dir}" -name "*.flac" | sort | while read -r filename; do
        id=$(basename ${filename%.flac})
        spk=$(basename $(dirname ${filename}))
        echo "${id} ${filename}" >> ${scp}
        echo "${id} ${spk}" >> ${utt2spk}
    done
    utils/utt2spk_to_spk2utt.pl ${utt2spk} > ${spk2utt}

    # make text using the original text
    # cleaning and phoneme conversion are performed on-the-fly during the training
    for f in downloads/datasets/gos/GroTTS/*.tsv; do
        # paste -d " " <(cut -f 1 < $f) <(cut -f 2 < $f) | local/clean_text.py >> ${text}
        paste -d " " <(cut -f 1 < $f) <(cut -f 2 < $f) >> ${text}
    done

    utils/validate_data_dir.sh --no-feats data/train
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: utils/subset_data_dir.sh"
    # development set
    utils/subset_data_dir.sh --per-spk data/train 50 data/train_dev

    # train set exluding development set
    mkdir -p data/train_nodev
    comm -23 data/train/utt2spk data/train_dev/utt2spk | cut -d " " -f 1 > data/train_nodev/utt
    utils/subset_data_dir.sh --utt-list data/train_nodev/utt data/train data/train_nodev
    rm data/train_nodev/utt
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
