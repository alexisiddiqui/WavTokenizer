# --coding:utf-8--
import json
import os
import subprocess
import time

import numpy as np
import torch
import torchaudio
from decoder.pretrained import WavTokenizer

print("Starting script execution...")


def convert_m4a_to_wav(input_path, output_path):
    """Convert M4A to WAV using ffmpeg"""
    try:
        command = [
            "ffmpeg",
            "-i",
            input_path,
            "-acodec",
            "pcm_s16le",
            "-ar",
            "24000",
            "-ac",
            "1",
            "-y",  # Overwrite output file if it exists
            output_path,
        ]
        subprocess.run(command, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
        return False
    except Exception as e:
        print(f"Conversion error: {str(e)}")
        return False


# Device setup
device1 = torch.device("cuda:0")
print(f"Using device: {device1}")

# Set up paths
base_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(base_dir)
print(f"Changed working directory to: {base_dir}")

# Input path points to m4a files directory
input_path = "/home/alexi/Documents/music-preview-visualiser/data/30000/previews"
out_folder = "/home/alexi/Documents/WavTokenizer/wavtokenizer/result/infer_m4a"
temp_wav_dir = os.path.join(out_folder, "temp_wav")
print(f"Input path: {input_path}")
print(f"Output folder: {out_folder}")

# Create necessary directories
ll = "wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn_testclean_epoch34"
tmptmp = out_folder + "/" + ll
features_dir = os.path.join(tmptmp, "intermediate_features")
os.makedirs(tmptmp, exist_ok=True)
os.makedirs(features_dir, exist_ok=True)
os.makedirs(temp_wav_dir, exist_ok=True)
print(f"Created output directories: {tmptmp}, {features_dir}")

# Benchmarking data structure
timing_stats = {
    "files": {},
    "summary": {
        "total_conversion_time": 0,
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

# Collect all M4A files recursively
print("\nCollecting M4A files...")
m4a_files = []
for root, dirs, files in os.walk(input_path):
    for file in files:
        if file.endswith(".m4a"):
            m4a_files.append(os.path.join(root, file))

print(f"Found {len(m4a_files)} M4A files")
print(f"First few files: {m4a_files[:3]}")

# Process all files
print("\nStarting encoding process...")
for i, m4a_path in enumerate(m4a_files):
    try:
        print(f"\nProcessing file {i + 1}/{len(m4a_files)}")
        print(f"Loading: {m4a_path}")

        file_timing = {"conversion_time": 0, "encoding_time": 0, "decoding_time": 0}
        timing_stats["files"][m4a_path] = file_timing

        # Convert M4A to WAV
        temp_wav_path = os.path.join(
            temp_wav_dir, f"{os.path.splitext(os.path.basename(m4a_path))[0]}.wav"
        )
        convert_start = time.time()
        if not convert_m4a_to_wav(m4a_path, temp_wav_path):
            print(f"Failed to convert {m4a_path} to WAV. Skipping...")
            continue
        convert_time = time.time() - convert_start
        file_timing["conversion_time"] = convert_time
        timing_stats["summary"]["total_conversion_time"] += convert_time

        # Load converted WAV file
        wav, sr = torchaudio.load(temp_wav_path)
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
        feature_filename = os.path.splitext(os.path.basename(m4a_path))[0] + "_features.npy"
        feature_path = os.path.join(features_dir, feature_filename)
        np.save(feature_path, features.cpu().numpy())
        print(f"Saved features to: {feature_path}")

        # Immediate decode and save
        print("\nStarting decoding...")
        decode_start = time.time()

        loaded_features = torch.from_numpy(np.load(feature_path)).to(device1)
        bandwidth_id = torch.tensor([0]).to(device1)

        audio_out = wavtokenizer.decode(loaded_features, bandwidth_id=bandwidth_id)
        decode_time = time.time() - decode_start
        file_timing["decoding_time"] = decode_time
        timing_stats["summary"]["total_decoding_time"] += decode_time
        print(f"Decode complete. Output audio shape: {audio_out.shape}")

        # Save as WAV directly
        output_filename = os.path.splitext(os.path.basename(m4a_path))[0] + ".wav"
        audio_path = os.path.join(tmptmp, output_filename)
        print(f"Saving to: {audio_path}")

        torchaudio.save(
            audio_path, audio_out.cpu(), sample_rate=24000, encoding="PCM_S", bits_per_sample=16
        )
        print("File saved successfully")
        timing_stats["summary"]["successful_files"] += 1

        # Clean up temporary WAV file
        os.remove(temp_wav_path)

    except Exception as e:
        print(f"Error processing file {m4a_path}: {str(e)}")
        continue

timing_stats["summary"]["total_files"] = len(m4a_files)

# Calculate and save timing statistics
if timing_stats["summary"]["successful_files"] > 0:
    avg_convert_time = (
        timing_stats["summary"]["total_conversion_time"]
        / timing_stats["summary"]["successful_files"]
    )
    avg_encode_time = (
        timing_stats["summary"]["total_encoding_time"] / timing_stats["summary"]["successful_files"]
    )
    avg_decode_time = (
        timing_stats["summary"]["total_decoding_time"] / timing_stats["summary"]["successful_files"]
    )

    timing_stats["summary"]["average_conversion_time"] = avg_convert_time
    timing_stats["summary"]["average_encoding_time"] = avg_encode_time
    timing_stats["summary"]["average_decoding_time"] = avg_decode_time
    timing_stats["summary"]["average_total_time"] = (
        avg_convert_time + avg_encode_time + avg_decode_time
    )

# Save timing statistics
timing_path = os.path.join(tmptmp, "processing_times.json")
with open(timing_path, "w") as f:
    json.dump(timing_stats, f, indent=4)

print("\nScript execution completed!")
print(
    f"Processed {timing_stats['summary']['successful_files']}/{timing_stats['summary']['total_files']} files successfully"
)
print(
    f"Average conversion time per file: {timing_stats['summary']['average_conversion_time']:.2f} seconds"
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

# Clean up temporary directory if empty
try:
    os.rmdir(temp_wav_dir)
except OSError:
    print(f"Note: Temporary WAV directory {temp_wav_dir} not empty or couldn't be removed")
