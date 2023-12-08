from argparse import ArgumentParser
import os
from espnet2.bin.tts_inference import Text2Speech
import soundfile as sf
import numpy as np


parser = ArgumentParser()
parser.add_argument("exp", type=str)
parser.add_argument("-c", "--checkpoint", type=str, default="exp/tts_vits/train.total_count.ave.pth")
parser.add_argument("-i", "--input", default="verhaaltjes/bragel.txt")
parser.add_argument("-o", "--output", default="examples")
args = parser.parse_args()

model_name = os.path.basename(args.exp)

model_file = os.path.join(args.exp, args.checkpoint)
check_name = os.readlink(model_file) if os.path.islink(model_file) else os.path.basename(model_file)
print(f"Processing: {check_name}")
epoch_name = check_name[:-4] if check_name.endswith('.pth') else check_name  # Remove '.pth' suffix

text2speech = Text2Speech(
    train_config=os.path.join(args.exp, "config.yaml"),
    model_file=model_file,
    speed_control_alpha=1.0,
    noise_scale=1.0,
    noise_scale_dur=1.0,
)

with open(args.input, "r") as f:
    lines = [line.strip().lower() for line in f.readlines()]

output_dir = os.path.join(args.output, os.path.basename(args.input)[:-4])

os.makedirs(output_dir, exist_ok=True)
for sid in [1, 2, 3]:
    wav_name = f"{model_name}-{sid}sid-{epoch_name}.wav"
    print(f"Generating {output_dir}/{wav_name}")

    outputs = []
    for line in lines:
        line = line.lower()
        outputs.append(text2speech(line, sids=np.array([sid])))
        
    speech = np.concatenate([output["wav"] for output in outputs])
    sf.write(os.path.join(output_dir, wav_name), speech, text2speech.fs)
