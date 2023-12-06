#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_config=conf/train_vits.yaml

init_param=downloads/espnet/kan-bayashi_ljspeech_vits/exp/tts_train_vits_raw_phn_tacotron_g2p_en_no_space/train.total_count.ave_10best.pth

lr=$(grep -A 10 'optim:' conf/train_vits.yaml | awk '/lr:/ {print $2; exit}')

./tts.sh \
    --lang gos \
    --feats_type raw \
    --audio_format flac \
    --fs 22050 \
    --n_fft 1024 \
    --n_shift 256 \
    --token_type char \
    --cleaner none \
    --g2p none \
    --use_sid true \
    --min_wav_duration 1 \
    --max_wav_duration 20 \
    --tts_task gan_tts \
    --train_config "${train_config}" \
    --train_set "train_nodev" \
    --valid_set "train_dev" \
    --test_sets "train_dev" \
    --srctexts "data/train/text" \
    --train_args "--use_wandb true --wandb_project GROTTS --wandb_name VITS_lr_${lr} --init_param ${init_param}:tts:tts:tts.generator.text_encoder,tts.generator.posterior_encoder.input_conv" \
    --expdir "exp-vits" \
    --tag "vits" \
    "$@"
