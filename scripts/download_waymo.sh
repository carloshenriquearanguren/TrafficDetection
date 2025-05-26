#!/usr/bin/env bash
set -e

# Example for a pip-installable demo package:
pip install waymo-open-dataset

mkdir -p datasets/waymo
cd datasets/waymo

# Replace with the actual demo‐download command you have:
waymo_download --output_dir .

echo "Waymo data ready in datasets/waymo/"
