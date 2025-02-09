# --coding:utf-8--
import json
import os
import time

import numpy as np
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
features_dir = os.path.join(tmptmp, "intermediate_features")
os.makedirs(tmptmp, exist_ok=True)
os.makedirs(features_dir, exist_ok=True)
print(f"Created output directories: {tmptmp}, {features_dir}")

# Benchmarking data structure
timing_stats = {
    "files": {},
    "summary": {
        "total_encoding_time": 0,
        "total_decoding_time": 0,
        "total_files": 0,
        "successful_files": 0,
    },
}

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
for i, wav_path in enumerate(wav_files):
    try:
        print(f"\nProcessing file {i + 1}/{len(wav_files)}")
        print(f"Loading: {wav_path}")

        file_timing = {"encoding_time": 0, "decoding_time": 0}
        timing_stats["files"][wav_path] = file_timing

        wav, sr = torchaudio.load(wav_path)
        print(f"Loaded audio shape: {wav.shape}, Sample rate: {sr}")

        bandwidth_id = torch.tensor([0])
        wav = wav.to(device1)
        print("Moved audio to GPU")

        # Encode and time the process
        print("Starting encoding...")
        encode_start = time.time()
        features, discrete_code = wavtokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)
        encode_time = time.time() - encode_start
        file_timing["encoding_time"] = encode_time
        timing_stats["summary"]["total_encoding_time"] += encode_time
        print(f"Encoding complete. Features shape: {features.shape}")

        # Save features to disk
        feature_filename = os.path.splitext(os.path.basename(wav_path))[0] + "_features.npy"
        feature_path = os.path.join(features_dir, feature_filename)
        np.save(feature_path, features.cpu().numpy())
        print(f"Saved features to: {feature_path}")

        # Immediate decode and save
        print("\nStarting decoding...")
        decode_start = time.time()

        # Load features from disk to simulate full pipeline
        loaded_features = torch.from_numpy(np.load(feature_path)).to(device1)
        bandwidth_id = torch.tensor([0]).to(device1)

        audio_out = wavtokenizer.decode(loaded_features, bandwidth_id=bandwidth_id)
        decode_time = time.time() - decode_start
        file_timing["decoding_time"] = decode_time
        timing_stats["summary"]["total_decoding_time"] += decode_time

        print(f"Decode complete. Output audio shape: {audio_out.shape}")

        # Save decoded audio
        output_filename = os.path.basename(wav_path)
        audio_path = os.path.join(tmptmp, output_filename)
        print(f"Saving to: {audio_path}")

        torchaudio.save(
            audio_path, audio_out.cpu(), sample_rate=24000, encoding="PCM_S", bits_per_sample=16
        )
        print("File saved successfully")

        timing_stats["summary"]["successful_files"] += 1

    except Exception as e:
        print(f"Error processing file {wav_path}: {str(e)}")
        continue

timing_stats["summary"]["total_files"] = len(wav_files)

# Calculate and save timing statistics
if timing_stats["summary"]["successful_files"] > 0:
    avg_encode_time = (
        timing_stats["summary"]["total_encoding_time"] / timing_stats["summary"]["successful_files"]
    )
    avg_decode_time = (
        timing_stats["summary"]["total_decoding_time"] / timing_stats["summary"]["successful_files"]
    )

    timing_stats["summary"]["average_encoding_time"] = avg_encode_time
    timing_stats["summary"]["average_decoding_time"] = avg_decode_time
    timing_stats["summary"]["average_total_time"] = avg_encode_time + avg_decode_time

# Save timing statistics
timing_path = os.path.join(tmptmp, "processing_times.json")
with open(timing_path, "w") as f:
    json.dump(timing_stats, f, indent=4)

print("\nScript execution completed!")
print(
    f"Processed {timing_stats['summary']['successful_files']}/{timing_stats['summary']['total_files']} files successfully"
)
print(
    f"Average encoding time per file: {timing_stats['summary']['average_encoding_time']:.2f} seconds"
)
print(
    f"Average decoding time per file: {timing_stats['summary']['average_decoding_time']:.2f} seconds"
)
print(
    f"Average total processing time per file: {timing_stats['summary']['average_total_time']:.2f} seconds"
)
print(f"Output files can be found in: {tmptmp}")
print(f"Intermediate features stored in: {features_dir}")
print(f"Detailed timing statistics saved to: {timing_path}")
