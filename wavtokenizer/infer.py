# --coding:utf-8--
import os

import torch
import torchaudio
from decoder.pretrained import WavTokenizer

print("Starting script execution...")

# Device setup
device1 = torch.device("cuda:0")
print(f"Using device: {device1}")

# Set up paths
base_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(base_dir)
print(f"Changed working directory to: {base_dir}")

# Input path points to LibriTTS test-clean directory
input_path = "/home/alexi/Documents/WavTokenizer/wavtokenizer/data/infer/lirbitts_testclean/LibriTTS/test-clean"
out_folder = "/home/alexi/Documents/WavTokenizer/wavtokenizer/result/infer"
print(f"Input path: {input_path}")
print(f"Output folder: {out_folder}")

# Model configuration
ll = "wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn_testclean_epoch34"
tmptmp = out_folder + "/" + ll
os.makedirs(tmptmp, exist_ok=True)
print(f"Created output directory: {tmptmp}")

# Load model
print("\nLoading model...")
config_path = "/home/alexi/Documents/WavTokenizer/wavtokenizer/configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
model_path = "/home/alexi/Documents/music-preview-visualiser/raw_data/WavTokenizer-large-speech-75token/wavtokenizer_large_speech_320_24k.ckpt"
print(f"Config path: {config_path}")
print(f"Model path: {model_path}")

try:
    wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
    wavtokenizer = wavtokenizer.to(device1)
    print("Model loaded successfully and moved to GPU")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

# Collect all WAV files recursively
print("\nCollecting WAV files...")
wav_files = []
for root, dirs, files in os.walk(input_path):
    for file in files:
        if file.endswith(".wav"):
            wav_files.append(os.path.join(root, file))

print(f"Found {len(wav_files)} WAV files")
print(f"First few files: {wav_files[:3]}")

# Process all files
print("\nStarting encoding process...")
features_all = []
for i, wav_path in enumerate(wav_files):
    try:
        print(f"\nProcessing file {i + 1}/{len(wav_files)}")
        print(f"Loading: {wav_path}")

        wav, sr = torchaudio.load(wav_path)
        print(f"Loaded audio shape: {wav.shape}, Sample rate: {sr}")

        bandwidth_id = torch.tensor([0])
        wav = wav.to(device1)
        print("Moved audio to GPU")

        print("Starting encoding...")
        features, discrete_code = wavtokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)
        print(f"Encoding complete. Features shape: {features.shape}")

        features_all.append(features)
        print("Features stored successfully")

    except Exception as e:
        print(f"Error processing file {wav_path}: {str(e)}")
        continue

# Decode and save all files
print("\nStarting decoding process...")
for i, wav_path in enumerate(wav_files):
    try:
        print(f"\nDecoding file {i + 1}/{len(wav_files)}")
        print(f"Processing: {wav_path}")

        bandwidth_id = torch.tensor([0]).to(device1)
        print("Starting decode...")
        audio_out = wavtokenizer.decode(features_all[i], bandwidth_id=bandwidth_id)
        print(f"Decode complete. Output audio shape: {audio_out.shape}")

        # Create output path
        output_filename = os.path.basename(wav_path)
        audio_path = os.path.join(tmptmp, output_filename)
        print(f"Saving to: {audio_path}")

        torchaudio.save(
            audio_path, audio_out.cpu(), sample_rate=24000, encoding="PCM_S", bits_per_sample=16
        )
        print("File saved successfully")

    except Exception as e:
        print(f"Error decoding/saving file {wav_path}: {str(e)}")
        continue

print("\nScript execution completed!")
print(f"Processed {len(wav_files)} files")
print(f"Output files can be found in: {tmptmp}")
