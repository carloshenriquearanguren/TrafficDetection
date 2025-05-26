#!/usr/bin/env bash
set -e

mkdir -p datasets/kitti
cd datasets/kitti

# Download and unzip KITTI detection
curl -L -O "http://www.cvlibs.net/download.php?file=data_object_image_2.zip"
unzip -o data_object_image_2.zip && rm data_object_image_2.zip

curl -L -O "http://www.cvlibs.net/download.php?file=data_object_label_2.zip"
unzip -o data_object_label_2.zip && rm data_object_label_2.zip

echo "KITTI data ready in datasets/kitti/{image_2,label_2}"
