from espnet2.bin.tts_inference import Text2Speech
import soundfile as sf
import numpy as np
import os

# Set your parameters
exp_dir = "./exp/tts_vits"
checkpoint = "latest.pth"  # Use the latest checkpoint
text = "Ze gingen mit klas noar Waddendiek, over en deur bragel lopen."
output_dir = "examples"  # Output directory for saving audio files

# Construct full path for the checkpoint
model_file = os.path.join(exp_dir, checkpoint)
check_name = os.readlink(model_file) if os.path.islink(model_file) else os.path.basename(model_file)
epoch_name = check_name[:-4] if check_name.endswith('.pth') else check_name  # Remove '.pth' suffix

# Initialize Text2Speech from the checkpoint
text2speech = Text2Speech(
    train_config=os.path.join(exp_dir, "config.yaml"),
    model_file=model_file,
)
print(text2speech.model)

# Create the output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Generate and save audio files for different speaker IDs (if applicable)
for sid in [1, 2, 3]:
    wav_name = f"{epoch_name}-{sid}sid.wav"
    print(f"Generating {wav_name}")
    output = text2speech(text.lower(), sids=np.array([sid]))
    sf.write(os.path.join(output_dir, wav_name), output["wav"], text2speech.fs)
