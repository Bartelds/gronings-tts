# TTS for Gronings
Train an end-to-end TTS system for Gronings using the [ESPnet toolkit](https://github.com/espnet/espnet).

> ESPnet is an end-to-end speech processing toolkit covering end-to-end speech recognition, text-to-speech, speech translation, speech enhancement, speaker diarization, spoken language understanding, and so on.
ESPnet uses [pytorch](http://pytorch.org/) as a deep learning engine and also follows [Kaldi](http://kaldi-asr.org/) style data processing, feature extraction/format, and recipes to provide a complete setup for various speech processing experiments.

This repository explains the use of VITS and is based on the [ESPnet documentation](https://github.com/espnet/espnet), but ESPnet also contains the following architectures (see [their repository](https://github.com/espnet/espnet) for examples):
- Tacotron2
- Transformer-TTS
- FastSpeech
- FastSpeech2
- Conformer FastSpeech & FastSpeech2
- JETS

## Step 0: Installation

Follow the steps described in the [ESPnet installation guide](https://espnet.github.io/espnet/installation.html) and skip step 1.

In summary (taken from the guide):

### Requirements
- Python 3.7+
- gcc 4.9+ for PyTorch 1.10.2+
- cmake3 for some extensions
- sox
- flac (This is not required when installing, but used in some recipes)

#### Installation Commands

Git clone ESPnet:
```bash
cd <any-place>
git clone https://github.com/espnet/espnet
```
Setup Python environment:
- You must create `<espnet-root>/tools/activate_python.sh` to specify the Python interpreter used in ESPnet recipes.
- To understand how ESPnet specifies Python, see `path.sh` for example.
```bash
cd <espnet-root>/tools
./setup_venv.sh $(command -v python3)
```

Install ESPnet:
- For CUDA version check `nvidia-smi` on the machine.
```bash
cd <espnet-root>/tools
make TH_VERSION=<PyTorch version> CUDA_VERSION=<CUDA version>
```

#### Check Installation
```bash
cd <espnet-root>/tools
bash -c ". ./activate_python.sh; . ./extra_path.sh; python3 check_install.py"
```

## Step 1: Data Preparation
KALDI-style data preparation is used and we follow the ESPnet folder hierarchy:
- Some (demo) files are available in this repository. 
```bash
cd <espnet-root>/egs2/grotts/tts1
```

#### Create Required Directories:

`data/`: The main directory for your data, which contains subdirectories for different data partitions.

- `data/train`: Contains the entire training dataset.
- `data/train_dev`: Includes a subset of train used for development purposes. This helps in validating the model during training.
- `data/train_nodev`: This is the training data excluding the development subset (`train_dev`). It's used for actual model training, ensuring that the development data is not seen during this phase.

#### Generate Necessary Files:
In each subdirectory (`data/train`, `data/train_dev`, and `data/train_nodev`), create the following files:

- `text`: Contains the transcriptions of each utterance (10 example lines available). Format: `<uttid> <transcription>`.
- `wav.scp`: Lists the audio file paths (10 example lines available). Format: `<uttid> /path/to/audio.wav`.
- `utt2spk`: Maps each utterance to its corresponding speaker. Format: `<uttid> <speakerid>`.
- `spk2utt`: Maps each speaker to their utterances. Format: `<speakerid> <uttid1> <uttid2> ...`.

#### Add Audio data:
Add audio files to a folder as specified in `wav.scp`.

See the `downloads` folder for an example.

The current setup shows the data prepared for three variants of Gronings, namely Hoogelaandsters, Oldambsters and Westerkertaaiers.

**New data or data from new variants can be added according to the setup explained above.**

#### [Optional] Automatically Generate Files Using Kaldi:
Install KALDI:
```bash
cd <kaldi-root>
git clone https://github.com/kaldi-asr/kaldi.git kaldi --origin upstream
cd kaldi
```
Follow the top-level installation instructions are in the file INSTALL.

- Automatically generate `spk2utt`: `utils/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt`.
  -  Repeat for `train_dev` and `train_nodev`.
- Create validation set (based on utterances): `utils/subset_data_dir.sh --utt-list data/train <number of utterances> data/train_dev`.
- Create validation set (based on speakers): `utils/subset_data_dir.sh --speakers data/train <number of speakers> data/train_dev`.
- Update training set (train_nodev) by exluding the validation set from `train_nodev`: `utils/fix_data_dir.sh data/train_nodev`.

## Step 2: Start Training
Start training the VITS model.

#### [Recommended] Setup WandB
Create an account at https://wandb.ai/site.

Login to your account on your machine:
```bash
wandb login
```

#### Start Training Procedure
The script below continues training from an [English VITS model](https://huggingface.co/espnet/kan-bayashi_ljspeech_vits) and has shown faster convergence in preliminary experiments:
- If you use this script **make sure all audio is sampled at 22kHz**.
- To train a VITS model from scratch, remove the reference to the English model in the script:
  - `downloads/espnet/kan-bayashi_ljspeech_vits/exp/tts_train_vits_raw_phn_tacotron_g2p_en_no_space/train.total_count.ave_10best.pth`.
```bash
./run_vits.sh
```

#### Change Parameters of the Model
The parameters as specified in `conf/train_vits.yaml` can be changed in `./run_vits.sh`.

For example, the batch size can be altered by modifying `--batch_bins <size>` in `--train_args` or the `audio_format` can be set by altering `audio_format`.

A modified script that uses 2 GPUs, an increase batch size, and WandB integration with project name `GROTTS` would look like this:
```bash
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
    --train_args "--use_wandb true --wandb_project GROTTS --wandb_name VITS_lr_${lr} --init_param ${init_param}:tts:tts:tts.generator.text_encoder,tts.generator.posterior_encoder.input_conv --batch_size 40 --batch_bins 10000000" \
    --expdir "exp-vits-lr-3e-4" \
    --tag "vits" \
    --ngpu 2 \
    "$@"
```

## Step 3: Generate Speech Using a Trained Model
To generate speech samples, use the following script:
```bash
python3 test.py
```

In this script set the following information:
```bash
exp_dir = "./exp/tts_vits" # Location where the TTS model is saved.
checkpoint = "latest.pth"  # Use the latest model checkpoint.
text = "Ze gingen mit klas noar Waddendiek, over en deur bragel lopen." # text to be used for generating speech.
output_dir = "examples"  # Output directory for saving audio files.
```

## Useful Links
- [Run ESPnet with your own corpus](https://github.com/espnet/espnet/tree/master/egs2/TEMPLATE#about-kaldi-style-data-directory)
- [ESPnet2 TTS Recipe TEMPLATE](https://github.com/espnet/espnet/blob/master/egs2/TEMPLATE/tts1/README.md#vits-training)
- [Older version of Gronings TTS recipe](https://github.com/samin9796/gro-tts/tree/main/egs2/gro_tts/tts1)