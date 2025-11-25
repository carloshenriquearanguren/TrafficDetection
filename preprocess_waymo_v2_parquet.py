import sys
print("--- Python sys.path ---")
for p in sys.path:
    print(p)
print("--- End Python sys.path ---")

try:
    import waymo_open_dataset
    print(f"Successfully imported 'waymo_open_dataset'. Version: {getattr(waymo_open_dataset, '__version__', 'N/A')}")
    print(f"Location: {waymo_open_dataset.__file__}")
    # Try to list contents of v2 if possible, though this might fail if v2 itself is the issue
    try:
        import waymo_open_dataset.v2
        print(f"Successfully imported 'waymo_open_dataset.v2'.")
        print(f"Location: {waymo_open_dataset.v2.__file__}")
    except ImportError as e_v2:
        print(f"Failed to import 'waymo_open_dataset.v2': {e_v2}")
except ImportError as e_top:
    print(f"Failed to import 'waymo_open_dataset' (top-level): {e_top}")

import os
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import io
import argparse
from tqdm import tqdm
import pandas as pd

# Waymo Open Dataset v2 imports
from waymo_open_dataset.v2 import (
    CameraImageComponent,
    CameraSegmentationLabelComponent
)
# Comment out or remove the problematic import
# from waymo_open_dataset.v2.dataset import read_components # This was line 35 before adding pandas import

# --- Configuration ---

# IMPORTANT: Update these GCS paths to your v2.0.1 dataset
# Example: gs://your-bucket-name/waymo_open_dataset_v_2_0_1/
GCS_DATASET_ROOT_PATH = "gs://waymo_open_dataset_v_2_0_1/"

# Original Waymo 2D PVPS class details (28 classes, 0-27 for semantic labels)
# We need to confirm these are the exact IDs used in the panoptic_label after `>> 8`
# For now, assuming these are the direct semantic IDs we get.
WAYMO_CLASS_NAMES_V2 = [
    "UNDEFINED",         # ID 0
    "CAR",               # ID 1
    "TRUCK",             # ID 2
    "BUS",               # ID 3
    "OTHER_VEHICLE",     # ID 4
    "MOTORCYCLIST",      # ID 5
    "BICYCLIST",         # ID 6
    "PEDESTRIAN",        # ID 7
    "SIGN",              # ID 8
    "TRAFFIC_LIGHT",     # ID 9
    "POLE",              # ID 10
    "CONSTRUCTION_CONE", # ID 11
    "BICYCLE",           # ID 12
    "MOTORCYCLE",        # ID 13
    "BUILDING",          # ID 14
    "VEGETATION",        # ID 15
    "TREE_TRUNK",        # ID 16
    "CURB",              # ID 17
    "ROAD",              # ID 18
    "LANE_MARKER",       # ID 19
    "OTHER_GROUND",      # ID 20
    "WALKABLE",          # ID 21
    "SIDEWALK",          # ID 22
    "SKY",               # ID 23
    "GROUND",            # ID 24
    "DYNAMIC",           # ID 25
    "STATIC",            # ID 26
    "BACKGROUND"         # ID 27
]
WAYMO_CLASS_IDS_V2 = list(range(len(WAYMO_CLASS_NAMES_V2)))


# --- Your Model's Class Configuration (Example for DeepLabv3 with Cityscapes-like classes) ---
# These are the classes your DeepLabv3 model is trained on or expects.
# Adjust these based on your actual model's requirements.
MODEL_TARGET_CLASS_NAMES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 
    'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 
    'motorcycle', 'bicycle', 'unlabeled'
]
NUM_MODEL_CLASSES = len(MODEL_TARGET_CLASS_NAMES)
MODEL_TARGET_CLASS_IDS = list(range(len(MODEL_TARGET_CLASS_NAMES)))
IGNORE_INDEX = MODEL_TARGET_CLASS_NAMES.index('unlabeled')  # = 19

# --- Mapping from Waymo v2 Semantic IDs to Your Model's Target IDs ---
# This is CRITICAL and needs to be accurate.
# Create a mapping: waymo_semantic_id -> your_model_target_id
# Example: If Waymo's "CAR" (ID 1) maps to your model's "car" (ID 13)
#          If Waymo's "ROAD" (ID 18) maps to your model's "road" (ID 0)
WAYMO_V2_TO_MODEL_ID_MAPPING = {
    # Waymo UNDEFINED (0)        → IGNORE
    0: IGNORE_INDEX,

    # Waymo CAR (1)              → your 'car' index (13)
    1: MODEL_TARGET_CLASS_NAMES.index('car'),         # =13

    # Waymo TRUCK (2)            → your 'truck' index (14)
    2: MODEL_TARGET_CLASS_NAMES.index('truck'),       # =14

    # Waymo BUS (3)              → your 'bus' index (15)
    3: MODEL_TARGET_CLASS_NAMES.index('bus'),         # =15

    # Waymo OTHER_VEHICLE (4)    → IGNORE (or map to 'car' if you want catch-all)
    4: IGNORE_INDEX,

    # Waymo MOTORCYCLIST (5)     → your 'rider' index (12)
    5: MODEL_TARGET_CLASS_NAMES.index('rider'),       # =12

    # Waymo BICYCLIST (6)        → your 'rider' index (12)
    6: MODEL_TARGET_CLASS_NAMES.index('rider'),       # =12

    # Waymo PEDESTRIAN (7)       → your 'person' index (11)
    7: MODEL_TARGET_CLASS_NAMES.index('person'),      # =11

    # Waymo SIGN (8)             → your 'traffic sign' index (7)
    8: MODEL_TARGET_CLASS_NAMES.index('traffic sign'),# =7

    # Waymo TRAFFIC_LIGHT (9)    → your 'traffic light' index (6)
    9: MODEL_TARGET_CLASS_NAMES.index('traffic light'),#=6

    # Waymo POLE (10)            → your 'pole' index (5)
    10: MODEL_TARGET_CLASS_NAMES.index('pole'),       # =5

    # Waymo CONSTRUCTION_CONE (11)→ IGNORE (or map to 'pole' if you prefer)
    11: IGNORE_INDEX,

    # Waymo BICYCLE (12)         → your 'bicycle' index (19)
    12: MODEL_TARGET_CLASS_NAMES.index('bicycle'),    # =19

    # Waymo MOTORCYCLE (13)      → your 'motorcycle' index (17)
    13: MODEL_TARGET_CLASS_NAMES.index('motorcycle'), # =17

    # Waymo BUILDING (14)        → your 'building' index (2)
    14: MODEL_TARGET_CLASS_NAMES.index('building'),   # =2

    # Waymo VEGETATION (15)      → your 'vegetation' index (8)
    15: MODEL_TARGET_CLASS_NAMES.index('vegetation'), # =8

    # Waymo TREE_TRUNK (16)      → IGNORE (or map to 'vegetation'/‘fence’ as you choose)
    16: IGNORE_INDEX,

    # Waymo CURB (17)           → your 'sidewalk' index (1)  # (Or map to IGNORE)
    17: MODEL_TARGET_CLASS_NAMES.index('sidewalk'),   # =1

    # Waymo ROAD (18)           → your 'road' index (0)
    18: MODEL_TARGET_CLASS_NAMES.index('road'),       # =0

    # Waymo LANE_MARKER (19)    → IGNORE (DeepLabv3 often doesn't have a separate lane line class)
    19: IGNORE_INDEX,

    # Waymo OTHER_GROUND (20)   → IGNORE
    20: IGNORE_INDEX,

    # Waymo WALKABLE (21)       → your 'sidewalk' index (1)
    21: MODEL_TARGET_CLASS_NAMES.index('sidewalk'),   # =1

    # Waymo SIDEWALK (22)       → your 'sidewalk' index (1)
    22: MODEL_TARGET_CLASS_NAMES.index('sidewalk'),   # =1

    # Waymo SKY (23)            → your 'sky' index (10)
    23: MODEL_TARGET_CLASS_NAMES.index('sky'),        # =10

    # Waymo GROUND (24)         → IGNORE (or map to 'road'/‘terrain’ if you have a terrain class)
    24: IGNORE_INDEX,

    # Waymo DYNAMIC (25)        → IGNORE
    25: IGNORE_INDEX,

    # Waymo STATIC (26)         → IGNORE
    26: IGNORE_INDEX,

    # Waymo BACKGROUND (27)     → IGNORE (= 19)
    27: IGNORE_INDEX
}

# (CHECK: Any waymo_id ∉ keys above → also set to IGNORE)
for waymo_id in range(28):
    if waymo_id not in WAYMO_V2_TO_MODEL_ID_MAPPING:
        WAYMO_V2_TO_MODEL_ID_MAPPING[waymo_id] = IGNORE_INDEX

def remap_semantic_ids(semantic_map_np, mapping_dict, ignore_index):
    """Remaps semantic IDs in a NumPy array based on a mapping dictionary."""
    remapped_mask = np.full_like(semantic_map_np, ignore_index, dtype=np.uint8)
    for waymo_id, model_id in mapping_dict.items():
        remapped_mask[semantic_map_np == waymo_id] = model_id
    return remapped_mask

def process_split(split_name, config, output_dataset_root):
    """Processes a given split (e.g., 'training', 'validation') of the Waymo v2 dataset."""

    print(f"Processing split: {split_name}")

    gcs_dataset_root_for_split = os.path.join(config['gcs_dataset_root'], split_name)
    camera_image_full_path = os.path.join(gcs_dataset_root_for_split, "camera_image")
    cam_seg_full_path = os.path.join(gcs_dataset_root_for_split, "camera_segmentation")

    image_save_dir = os.path.join(output_dataset_root, split_name, "images")
    mask_save_dir = os.path.join(output_dataset_root, split_name, "masks")
    os.makedirs(image_save_dir, exist_ok=True)
    os.makedirs(mask_save_dir, exist_ok=True)

    manifest_data = []

    print(f"Attempting to read Parquet components for {split_name} from:")
    print(f"Images: {camera_image_full_path}")
    print(f"Segmentations: {cam_seg_full_path}")

    try:
        # Ensure gcsfs is installed for pandas to read from GCS: pip install gcsfs
        # Also ensure GCS authentication is set up.
        print("Reading camera image Parquet files...")
        cam_img_df = pd.read_parquet(camera_image_full_path, engine='pyarrow')
        print(f"Camera Image DataFrame loaded. Shape: {cam_img_df.shape}")
        print("Columns in Camera Image DataFrame:", cam_img_df.columns.tolist())
        if cam_img_df.empty:
            print(f"Warning: Camera Image DataFrame is empty for {split_name}.")
            return
        print("First 5 rows of Camera Image DataFrame:")
        print(cam_img_df.head())

        print("\nReading camera segmentation Parquet files...")
        cam_seg_df = pd.read_parquet(cam_seg_full_path, engine='pyarrow')
        print(f"Camera Segmentation DataFrame loaded. Shape: {cam_seg_df.shape}")
        print("Columns in Camera Segmentation DataFrame:", cam_seg_df.columns.tolist())
        if cam_seg_df.empty:
            print(f"Warning: Camera Segmentation DataFrame is empty for {split_name}.")
            return
        print("First 5 rows of Camera Segmentation DataFrame:")
        print(cam_seg_df.head())

    except Exception as e:
        print(f"Error reading Parquet files from GCS for split {split_name}: {e}")
        print("Please ensure:")
        print("1. The GCS paths are correct and you have read access.")
        print("2. You have 'gcsfs' installed ('pip install gcsfs pyarrow').")
        print("3. Your GCS authentication is correctly set up (e.g., GOOGLE_APPLICATION_CREDENTIALS or gcloud auth).")
        return

    # --- MERGE DATAFRAMES --- 
    # We need to identify the correct key columns from the printouts above.
    # Common keys are usually related to: context_name, timestamp_micros, camera_name
    # Example key names (VERIFY THESE WITH THE ACTUAL COLUMN NAMES PRINTED ABOVE):
    key_columns = [
        'key.segment_context_name', 
        'key.frame_timestamp_micros',
        'key.camera_name'
    ]
    
    # Check if these example key columns exist in both dataframes
    missing_keys_img = [col for col in key_columns if col not in cam_img_df.columns]
    missing_keys_seg = [col for col in key_columns if col not in cam_seg_df.columns]

    if missing_keys_img:
        print(f"ERROR: Missing key columns in image DataFrame: {missing_keys_img}. Available: {cam_img_df.columns.tolist()}")
        return
    if missing_keys_seg:
        print(f"ERROR: Missing key columns in segmentation DataFrame: {missing_keys_seg}. Available: {cam_seg_df.columns.tolist()}")
        return
        
    print(f"\nAttempting to merge DataFrames on keys: {key_columns}")
    try:
        # Using a copy to avoid SettingWithCopyWarning if we modify merged_df later
        merged_df = pd.merge(cam_img_df, cam_seg_df, on=key_columns, how='inner').copy()
        print(f"Merged DataFrame shape: {merged_df.shape}")
        if merged_df.empty:
            print(f"Warning: Merged DataFrame is empty for {split_name}. This might indicate a mismatch in keys or no overlapping data.")
            print(f"Sample keys from cam_img_df ({key_columns[0]}):\n{cam_img_df[key_columns[0]].unique()[:5]}")
            print(f"Sample keys from cam_seg_df ({key_columns[0]}):\n{cam_seg_df[key_columns[0]].unique()[:5]}")
            return
        print("First 5 rows of Merged DataFrame:")
        print(merged_df.head())
    except Exception as e:
        print(f"Error merging DataFrames: {e}")
        print("Please verify the key_columns based on the printed column lists from each DataFrame.")
        return

    print(f"\nStarting iteration through merged data for {split_name} split...")
    frame_count = 0

    # We need to identify the actual column names for image bytes and panoptic label bytes
    # from the merged_df.columns.tolist() - e.g. '[CameraImageComponent].image'
    # For now, let's assume placeholder names and then adjust based on actual column names.
    # Example placeholder column names (VERIFY THESE!):
    image_bytes_col = '[CameraImageComponent].image' # Placeholder
    panoptic_label_bytes_col = '[CameraSegmentationLabelComponent].panoptic_label_divisor_256' # Placeholder for semantic
    panoptic_label_full_bytes_col = '[CameraSegmentationLabelComponent].panoptic_label' # Placeholder for full panoptic

    # Verify actual column names exist
    if image_bytes_col not in merged_df.columns:
        print(f"ERROR: Assumed image_bytes_col '{image_bytes_col}' not in merged_df. Columns: {merged_df.columns.tolist()}")
        return
    # Check for at least one of the panoptic label columns
    has_semantic_label_col = panoptic_label_bytes_col in merged_df.columns
    has_full_panoptic_label_col = panoptic_label_full_bytes_col in merged_df.columns
    if not has_semantic_label_col and not has_full_panoptic_label_col:
        print(f"ERROR: Neither assumed panoptic label column ('{panoptic_label_bytes_col}' or '{panoptic_label_full_bytes_col}') found in merged_df. Columns: {merged_df.columns.tolist()}")
        return

    for index, row in tqdm(merged_df.iterrows(), total=merged_df.shape[0], desc=f"Processing {split_name}"):
        # Extract keys from the row using the verified key_columns
        segment_id = row[key_columns[0]]
        timestamp_micros = row[key_columns[1]]
        # camera_name might need .name if it was an enum object before, but from pandas it should be a string/int
        camera_name = str(row[key_columns[2]]) 

        # Decode image
        try:
            img_bytes = row[image_bytes_col]
            if img_bytes is None or (isinstance(img_bytes, float) and np.isnan(img_bytes)): # Check for pd.NA or NaN
                print(f"Warning: Image bytes are None/NaN for {segment_id}, {timestamp_micros}, {camera_name}. Skipping.")
                continue
            pil_image = Image.open(io.BytesIO(img_bytes))
            image_np = np.array(pil_image) 
            if image_np.ndim != 3 or image_np.shape[2] != 3:
                print(f"Warning: Decoded image has unexpected shape {image_np.shape} for {segment_id}, {timestamp_micros}, {camera_name}. Skipping.")
                continue
        except Exception as e:
            print(f"Error decoding image for {segment_id}, {timestamp_micros}, {camera_name}: {e}. Skipping.")
            continue

        # Decode panoptic label
        try:
            panoptic_bytes = None
            is_direct_semantic = False
            if has_semantic_label_col:
                panoptic_bytes = row[panoptic_label_bytes_col]
                is_direct_semantic = True 
            elif has_full_panoptic_label_col:
                panoptic_bytes = row[panoptic_label_full_bytes_col]
                is_direct_semantic = False
            
            if panoptic_bytes is None or (isinstance(panoptic_bytes, float) and np.isnan(panoptic_bytes)):
                print(f"Warning: Panoptic label bytes are None/NaN for {segment_id}, {timestamp_micros}, {camera_name}. Skipping.")
                continue

            if is_direct_semantic:
                 semantic_map_flat = tf.io.decode_raw(panoptic_bytes, out_type=tf.uint8)
            else: 
                 panoptic_flat = tf.io.decode_raw(panoptic_bytes, out_type=tf.uint16)
                 panoptic_map_tf = tf.reshape(panoptic_flat, [image_np.shape[0], image_np.shape[1]])
                 semantic_map_flat = tf.cast(panoptic_map_tf // 256, tf.uint8)

            semantic_map_np = tf.reshape(semantic_map_flat, [image_np.shape[0], image_np.shape[1]]).numpy()

        except Exception as e:
            print(f"Error decoding panoptic label for {segment_id}, {timestamp_micros}, {camera_name}: {e}. Skipping.")
            continue

        # Remap semantic IDs
        remapped_mask_np = remap_semantic_ids(semantic_map_np, config['waymo_to_model_mapping'], config['ignore_index'])

        # Save image and mask
        image_filename = f"{segment_id}_{timestamp_micros}_{camera_name}.jpg"
        mask_filename = f"{segment_id}_{timestamp_micros}_{camera_name}_mask.png"

        try:
            Image.fromarray(image_np).save(os.path.join(image_save_dir, image_filename))
            Image.fromarray(remapped_mask_np).save(os.path.join(mask_save_dir, mask_filename))
        except Exception as e:
            print(f"Error saving image/mask for {image_filename}: {e}. Skipping.")
            continue

        manifest_data.append({
            "image_path": os.path.join(split_name, "images", image_filename),
            "mask_path": os.path.join(split_name, "masks", mask_filename),
            "original_segment_id": segment_id,
            "original_timestamp_micros": timestamp_micros,
            "original_camera_name": camera_name
        })
        frame_count += 1

    print(f"Processed and saved {frame_count} frames for split {split_name}.")

    manifest_path = os.path.join(output_dataset_root, f"{split_name}_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest_data, f, indent=4)
    print(f"Manifest file saved to {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess Waymo Open Dataset v2 Parquet data for semantic segmentation.")
    parser.add_argument("--gcs_dataset_root", type=str, default=None,
                        help="Root GCS path to the Waymo v2.0.1 dataset (e.g., gs://bucket/waymo_open_dataset_v_2_0_1/). This is REQUIRED.")
    parser.add_argument("--output_dataset_root", type=str, required=True,
                        help="Local directory to save the processed images, masks, and manifest files.")
    parser.add_argument("--splits", type=str, default="training,validation",
                        help="Comma-separated list of splits to process (e.g., 'training,validation').")

    args = parser.parse_args()

    # Prioritize command-line argument for gcs_dataset_root
    current_gcs_dataset_root = args.gcs_dataset_root

    # If --gcs_dataset_root is not provided, check the global variable as a fallback
    if current_gcs_dataset_root is None:
        print("Info: --gcs_dataset_root argument not provided. Checking global GCS_DATASET_ROOT_PATH variable.")
        current_gcs_dataset_root = GCS_DATASET_ROOT_PATH
    
    # Check if the determined GCS path is still the placeholder
    if current_gcs_dataset_root is None or "<YOUR_BUCKET_AND_PATH_TO_V2.0.1_DATASET>" in current_gcs_dataset_root or "YOUR_BUCKET_NAME" in current_gcs_dataset_root:
        parser.error("A valid GCS dataset root path must be provided either via the --gcs_dataset_root argument or by editing the GCS_DATASET_ROOT_PATH variable in the script. The current path is a placeholder.")

    print(f"Using GCS Dataset Root: {current_gcs_dataset_root}")

    config = {
        "gcs_dataset_root": current_gcs_dataset_root, # Use the determined path
        "waymo_to_model_mapping": WAYMO_V2_TO_MODEL_ID_MAPPING,
        "ignore_index": IGNORE_INDEX,
        "num_model_classes": NUM_MODEL_CLASSES
    }

    splits_to_process = [s.strip() for s in args.splits.split(',')]

    for split in splits_to_process:
        if split not in ["training", "validation"]: # Add "testing" if you have it and want to process it
            print(f"Warning: Unknown split '{split}' found in --splits argument. Skipping.")
            continue
        process_split(split, config, args.output_dataset_root)

    print("Preprocessing complete.")

if __name__ == "__main__":
    # It's good practice to ensure TensorFlow is using the GPU if available,
    # though for this script, it's mostly I/O and NumPy/PIL ops after decoding.
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"TensorFlow is using GPU(s): {gpus}")
        except RuntimeError as e:
            print(f"RuntimeError in GPU setup: {e}")
    else:
        print("TensorFlow is using CPU.")
    main() 