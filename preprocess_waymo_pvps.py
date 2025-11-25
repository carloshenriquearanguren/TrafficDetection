import tensorflow as tf # For TFRecordDataset
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils
# from waymo_open_dataset.utils import camera_utils # For image extraction - Removed as unused and causing import error
# label_utils.get_camera_labels was mentioned, but it seems to be part of camera_utils or a more general util now.
# The actual function from Waymo\'s more recent tutorials might be slightly different,
# e.g. directly accessing frame.camera_labels or frame.images[x].camera_segmentation_label
# We\'ll use the concept you provided: panoptic_label = get_camera_panoptic_label(frame, camera_name)

import numpy as np
import os
import json
from PIL import Image # For saving images and masks
import glob # For finding TFRecord files
import io # For BytesIO

# --- Configuration ---

# 1. Waymo 2D PVPS Class List & Original IDs (as provided)
WAYMO_CLASS_NAMES = {
    0: "UNKNOWN", 1: "DRIVABLE_SURFACE", 2: "LANE_DIVIDER", 3: "CROSSWALK",
    4: "STOP_LINE", 5: "SIDEWALK", 6: "CURB", 7: "SHOULDER",
    8: "BICYCLE_PATH", 9: "RAIL_TRACK", 10: "TRAILER", 11: "TRUCK",
    12: "BUS", 13: "TAXI", 14: "CAR", 15: "MOTORCYCLIST",
    16: "MOTORCYCLE", 17: "BICYCLE", 18: "PEDESTRIAN", 19: "ANIMAL",
    20: "SIGN", 21: "POLE", 22: "TRAFFIC_LIGHT", 23: "UTILITY_POLE",
    24: "WALL", 25: "FENCE", 26: "BUILDING", 27: "VEGETATION"
}

# 2. Chosen Target Classes (Original Waymo IDs) - MODIFY THIS LIST AS NEEDED
CHOSEN_ORIGINAL_IDS = sorted([
    1, 2, 3, 4, 5, 6,  # drivable + sidewalk
    10, 11, 12, 13, 14, # vehicles (car, truck, bus, trailer, taxi)
    15, 16, 17, 18,     # two-wheelers, pedestrian (motorcyclist, motorcycle, bicycle, pedestrian)
    # 19, # ANIMAL - decide if you want this
    20, 21, 22,         # sign, pole, traffic_light
    # 23, # UTILITY_POLE - often similar to POLE
    # 24, 25, # WALL, FENCE - decide
    26, 27,             # building, vegetation
    # 0 # UNKNOWN - usually mapped to ignore
])
# Filter out any IDs not in WAYMO_CLASS_NAMES if necessary (e.g. if a typo in CHOSEN_ORIGINAL_IDS)
CHOSEN_ORIGINAL_IDS = [id for id in CHOSEN_ORIGINAL_IDS if id in WAYMO_CLASS_NAMES]


# 3. Mapping Original Waymo IDs to New 0-Indexed Labels
ORIG_ID_TO_NEW_ID = {orig_id: new_id for new_id, orig_id in enumerate(CHOSEN_ORIGINAL_IDS)}
NUM_CLASSES_MODEL = len(CHOSEN_ORIGINAL_IDS) # Number of classes for your DeepLabv3
IGNORE_INDEX = 255

print(f"Number of target classes for model: {NUM_CLASSES_MODEL}")
print("Mapping from original Waymo ID to new 0-indexed ID:")
for orig_id in CHOSEN_ORIGINAL_IDS:
    print(f"  Waymo ID {orig_id} (\'{WAYMO_CLASS_NAMES[orig_id]}\') --> New ID {ORIG_ID_TO_NEW_ID[orig_id]}")


# 4. Input Data Path (Points to directory of TFRecord files or GCS prefix)
# Example: INPUT_WAYMO_ROOT_PATH = "/mnt/data/waymo/2d_pvps_tfrecord_subset/"
# Example: INPUT_WAYMO_ROOT_PATH = "gs://waymo-open-dataset/v1.4.2/camera_segmentation/" # Check exact GCS path for 2D PVPS TFRecords
# INPUT_WAYMO_ROOT_PATH = "./waymo_pvps_tfrecords/" # MODIFY THIS - Defaulting to a local subdirectory

# New: Define a list of potential GCS path configurations to try
POTENTIAL_GCS_PATH_CONFIGS = [
    {
        "description": "Focused Waymo v2.0.1 Camera Segmentation Path",
        "paths": {
            "train": "gs://waymo_open_dataset_v_2_0_1/training/camera_segmentation/",
            "val": "gs://waymo_open_dataset_v_2_0_1/validation/camera_segmentation/",
            # Add "test": "gs://waymo_open_dataset_v_2_0_1/testing/camera_segmentation/" if applicable
        }
    }
    # {
    #     "description": "Direct camera_segmentation folders v1.4.3 (e.g., training/camera_segmentation/)",
    #     "paths": {
    #         "train": "gs://waymo_open_dataset_v_1_4_3/training/camera_segmentation/",
    #         "val": "gs://waymo_open_dataset_v_1_4_3/validation/camera_segmentation/"
    #     }
    # },
    # {
    #     "description": "Individual Files v1.4.3 (direct training/validation)",
    #     "paths": {
    #         "train": "gs://waymo_open_dataset_v_1_4_3/individual_files/training/",
    #         "val": "gs://waymo_open_dataset_v_1_4_3/individual_files/validation/"
    #     }
    # },
    # {
    #     "description": "Archived Files - Domain Adaptation v1.4.3 (training/validation)",
    #     "paths": {
    #         "train": "gs://waymo_open_dataset_v_1_4_3/archived_files/domain_adaptation/training/",
    #         "val": "gs://waymo_open_dataset_v_1_4_3/archived_files/domain_adaptation/validation/"
    #     }
    # },
    # {
    #     "description": "Archived Files v1.4.3 (direct training/validation)",
    #     "paths": {
    #         "train": "gs://waymo_open_dataset_v_1_4_3/archived_files/training/",
    #         "val": "gs://waymo_open_dataset_v_1_4_3/archived_files/validation/"
    #     }
    # },
    # {
    #     "description": "Individual Files - Domain Adaptation v1.4.3 (training/validation)",
    #     "paths": {
    #         "train": "gs://waymo_open_dataset_v_1_4_3/individual_files/domain_adaptation/training/",
    #         "val": "gs://waymo_open_dataset_v_1_4_3/individual_files/domain_adaptation/validation/"
    #     }
    # }
]
# Ensure these GCS paths end with a trailing slash '/'

# 5. Output Data Path
# Example: OUTPUT_DATASET_ROOT = "/mnt/data/waymo_pvps_for_deeplab/"
OUTPUT_DATASET_ROOT = "./waymo_pvps_for_deeplab/" # MODIFY THIS - Defaulting to a local subdirectory

IMAGE_SAVE_DIR = os.path.join(OUTPUT_DATASET_ROOT, "images")
MASK_SAVE_DIR = os.path.join(OUTPUT_DATASET_ROOT, "masks")
MANIFEST_FILE_PATH = os.path.join(OUTPUT_DATASET_ROOT, "manifest.json")

# --- Helper Functions ---

def get_split_name(tfrecord_path): # Argument changed for clarity
    """
    Determines if a segment belongs to train, val, or test based on the input path.
    """
    path_lower = str(tfrecord_path).lower() # Ensure it's a string for path operations
    if "/validation/" in path_lower:
        return "val"
    elif "/training/" in path_lower:
        return "train"
    elif "/testing/" in path_lower: # If you add a test set
        return "test"
    else:
        # Fallback for local paths or if GCS path doesn't contain split name directly
        # This part might need adjustment if you use local paths without "training/" or "validation/" in them.
        # For the GCS structure provided by the user, the above checks should suffice.
        # If frame.context.name was intended for split detection from metadata, that's a different logic.
        # For now, we rely on the input path containing the split name.
        
        # Attempt to infer from basename if path doesn't contain split folder names
        basename_lower = os.path.basename(path_lower)
        if "val" in basename_lower: # Check for "val" in filename itself
             return "val"
        # if "train" in basename_lower: # Could add this but "/training/" in path is more robust
        #     return "train"

        print(f"Warning: Could not determine split from path '{tfrecord_path}'. Defaulting to 'train'. Review this behavior.")
        return "train" # Default, ensure this is desired if no other indicator is found


# --- Main Preprocessing Logic ---
def main():
    os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
    os.makedirs(MASK_SAVE_DIR, exist_ok=True)
    os.makedirs(os.path.join(IMAGE_SAVE_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(IMAGE_SAVE_DIR, "val"), exist_ok=True)
    os.makedirs(os.path.join(MASK_SAVE_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(MASK_SAVE_DIR, "val"), exist_ok=True)

    global_sample_id_counter = 0
    processed_successfully = False

    for config_idx, path_config_entry in enumerate(POTENTIAL_GCS_PATH_CONFIGS):
        print(f"\\n--- Attempting GCS Path Configuration {config_idx + 1}/{len(POTENTIAL_GCS_PATH_CONFIGS)} ---")
        print(f"Description: {path_config_entry['description']}")
        current_input_waymo_paths_by_split = path_config_entry['paths']
        
        manifest_data = {"train": [], "val": []} # Reset for each attempt
        # global_sample_id_counter = 0 # Reset counter for each top-level attempt

        all_tfrecord_files_with_split_info = []

        for split_type, input_path_root in current_input_waymo_paths_by_split.items():
            if not input_path_root: # Skip if a path for a split is not defined
                print(f"No input path defined for split '{split_type}', skipping.")
                continue

            print(f"Searching for TFRecord files for split '{split_type}' in: {input_path_root}")
            
            # Modified glob_pattern to look one level deeper
            # Original: glob_pattern = os.path.join(input_path_root, "*.tfrecord")
            glob_pattern = os.path.join(input_path_root, "*/*.tfrecord") # Look in subdirectories
            print(f"Using glob pattern: {glob_pattern}") # Debug print for the pattern

            try:
                current_split_tfrecords = tf.io.gfile.glob(glob_pattern)
            except Exception as e:
                print(f"Error using tf.io.gfile.glob for pattern {glob_pattern}: {e}")
                current_split_tfrecords = []

            if not current_split_tfrecords and '*' not in input_path_root and '?' not in input_path_root:
                 if tf.io.gfile.exists(input_path_root) and not tf.io.gfile.isdir(input_path_root) and input_path_root.endswith(".tfrecord"):
                     current_split_tfrecords = [input_path_root]

            if not current_split_tfrecords:
                print(f"No TFRecord files found in {input_path_root} (using pattern: {glob_pattern}) for split '{split_type}'.")
            else:
                print(f"Found {len(current_split_tfrecords)} TFRecord files in {input_path_root} for split '{split_type}'.")
                for tf_file in current_split_tfrecords:
                    all_tfrecord_files_with_split_info.append({"path": tf_file, "split_from_source": split_type})
        
        if not all_tfrecord_files_with_split_info:
            print(f"Configuration '{path_config_entry['description']}' yielded no TFRecord files. Trying next configuration.")
            continue

        print(f"Found a total of {len(all_tfrecord_files_with_split_info)} TFRecord files to process for configuration '{path_config_entry['description']}'.")
        
        # Reset sample counter for this specific configuration attempt
        current_config_samples_processed = 0

        for file_idx, file_info in enumerate(all_tfrecord_files_with_split_info):
            tfrecord_path = file_info["path"]
            # ... (rest of the inner loop for processing frames, images, masks)
            # Important: inside the innermost loop where a sample is successfully saved:
            # current_config_samples_processed += 1
            # global_sample_id_counter +=1
            # Ensure this increment happens:
            # if camera_image_pil is not None and panoptic_label_map_uint16 is not None:
            #   ... save image and mask ...
            #   manifest_data[split_for_saving].append(...)
            #   current_config_samples_processed += 1
            #   global_sample_id_counter +=1

# --- The existing frame processing loop needs to be nested here ---
# --- For brevity, I'll show where to integrate the sample increment ---
            split_for_saving = get_split_name(tfrecord_path)

            if split_for_saving not in manifest_data:
                manifest_data[split_for_saving] = [] # Ensure manifest_data is initialized for the split
                # Create directories if they don't exist for the current split_for_saving
                os.makedirs(os.path.join(IMAGE_SAVE_DIR, split_for_saving), exist_ok=True)
                os.makedirs(os.path.join(MASK_SAVE_DIR, split_for_saving), exist_ok=True)


            print(f"Processing TFRecord {file_idx+1}/{len(all_tfrecord_files_with_split_info)}: {tfrecord_path} for split '{split_for_saving}'...")
            segment_identifier = os.path.splitext(os.path.basename(tfrecord_path))[0]

            dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')
            
            frames_processed_in_current_tfrecord = 0 

            for frame_idx, data in enumerate(dataset):
                if frames_processed_in_current_tfrecord >= 2 and file_idx > 0: 
                    break
                if frames_processed_in_current_tfrecord >= 5 and file_idx == 0: 
                    print(f"Debug: Limiting to 5 frames for the first TFRecord ({tfrecord_path})...")
                    break
                
                frame = open_dataset.Frame()
                frame.ParseFromString(bytearray(data.numpy()))
                frames_processed_in_current_tfrecord += 1
                
                current_segment_name_from_frame = frame.context.name if frame.context and frame.context.name else segment_identifier
                if file_idx == 0 and frame_idx < 2: 
                    print(f"  Frame {frame_idx}: Context Name: {frame.context.name if frame.context else 'N/A'}")
                    print(f"    Number of camera_labels: {len(frame.camera_labels)}")
                    print(f"    Number of images: {len(frame.images)}")

                for camera_name_str, camera_name_enum_val in open_dataset.CameraName.Name.items():
                    if camera_name_enum_val == open_dataset.CameraName.UNKNOWN:
                        continue

                    if file_idx == 0 and frame_idx < 2: 
                        print(f"    Processing Camera: {camera_name_str} (Enum: {camera_name_enum_val})")

                    panoptic_label_map_uint16 = None
                    found_matching_camera_label = False
                    for cl_idx, cl in enumerate(frame.camera_labels): 
                        if cl.name == camera_name_enum_val:
                            found_matching_camera_label = True
                            if file_idx == 0 and frame_idx < 2: 
                                print(f"      Attempt 1: Checking cl (CameraLabel) for {camera_name_str}. Name: {cl.name}")
                                if hasattr(cl, 'panoptic_label'):
                                    print(f"        cl.panoptic_label exists. Is it populated? {bool(cl.panoptic_label)}")
                                    if cl.panoptic_label:
                                        print(f"        Type of cl.panoptic_label: {type(cl.panoptic_label)}, Length (if bytes/str): {len(cl.panoptic_label) if isinstance(cl.panoptic_label, (bytes, str)) else 'N/A'}")
                                else:
                                    print(f"        cl.panoptic_label does NOT exist on CameraLabel object.")
                            
                            if hasattr(cl, 'panoptic_label') and cl.panoptic_label:
                                try:
                                    panoptic_label_tensor = tf.io.decode_png(cl.panoptic_label, channels=1, dtype=tf.uint16)
                                    panoptic_label_map_uint16 = np.squeeze(panoptic_label_tensor.numpy(), axis=-1)
                                    if file_idx == 0 and frame_idx < 2: print(f"        Successfully decoded panoptic_label from cl for {camera_name_str}")
                                except Exception as e:
                                    print(f"        Error decoding panoptic_label from cl for camera {camera_name_str}: {e}")
                    
                    if panoptic_label_map_uint16 is None: 
                        if file_idx == 0 and frame_idx < 2: 
                            print(f"      Attempt 1 (cl.panoptic_label) failed for {camera_name_str}. Proceeding to Attempt 2 (img_data.camera_segmentation_label)...")
                        for img_data in frame.images:
                            if img_data.name == camera_name_enum_val:
                                if file_idx == 0 and frame_idx < 2: 
                                    print(f"        Found matching Image (img_data) for {camera_name_str}. Name: {img_data.name}")
                                    if hasattr(img_data, 'camera_segmentation_label'):
                                        print(f"          img_data.camera_segmentation_label exists.")
                                        seg_label_obj = img_data.camera_segmentation_label
                                        if hasattr(seg_label_obj, 'panoptic_label') and seg_label_obj.panoptic_label:
                                            print(f"            seg_label_obj.panoptic_label exists and is populated. Type: {type(seg_label_obj.panoptic_label)}, Length: {len(seg_label_obj.panoptic_label) if isinstance(seg_label_obj.panoptic_label, (bytes,str)) else 'N/A'}")
                                            try:
                                                panoptic_label_tensor = tf.io.decode_png(seg_label_obj.panoptic_label, channels=1, dtype=tf.uint16)
                                                panoptic_label_map_uint16 = np.squeeze(panoptic_label_tensor.numpy(), axis=-1)
                                                print(f"            Successfully decoded panoptic_label from img_data.camera_segmentation_label for {camera_name_str}")
                                            except Exception as e:
                                                print(f"            Error decoding panoptic_label from img_data.camera_segmentation_label for {camera_name_str}: {e}")
                                        else:
                                            print(f"            img_data.camera_segmentation_label.panoptic_label does NOT exist or is not populated.")
                                    else:
                                        print(f"          img_data.camera_segmentation_label does NOT exist.")
                                break 

                    if panoptic_label_map_uint16 is None:
                        if file_idx == 0 and frame_idx < 2 : 
                            print(f"      Both attempts failed to get a valid panoptic_label_map_uint16 for camera {camera_name_str}.")
                        continue

                    semantic_map_orig_ids = panoptic_label_map_uint16 // 256
                    H, W = semantic_map_orig_ids.shape
                    new_mask_for_model = np.full((H, W), IGNORE_INDEX, dtype=np.uint8)
                    for original_waymo_id, new_model_id in ORIG_ID_TO_NEW_ID.items():
                        new_mask_for_model[semantic_map_orig_ids == original_waymo_id] = new_model_id
                    
                    camera_image_pil = None
                    found_matching_image = False
                    for img_idx, img_data in enumerate(frame.images):
                        if img_data.name == camera_name_enum_val:
                            found_matching_image = True
                            if file_idx == 0 and frame_idx < 2: 
                                print(f"      Found matching image (img_idx {img_idx}) for {camera_name_str}. Name: {img_data.name}. Image bytes length: {len(img_data.image)}")
                            try:
                                pil_image = Image.open(io.BytesIO(img_data.image))
                                camera_image_pil = pil_image.convert('RGB')
                            except Exception as e:
                                print(f"Error opening image for camera {camera_name_str} in segment {current_segment_name_from_frame}, frame {frame_idx}: {e}")
                                camera_image_pil = None 
                            break

                    if camera_image_pil is None:
                        if file_idx == 0 and frame_idx < 2: 
                            if found_matching_image:
                                print(f"      Image data for {camera_name_str} was found but could not be opened/converted by PIL.")
                            else:
                                print(f"      No matching image data found in frame.images for camera {camera_name_str}.")
                        continue

                    if file_idx == 0 and frame_idx < 2: 
                        print(f"      SUCCESS: Got image and mask for {camera_name_str}. Proceeding to save.")

                    # Ensure split_for_saving (derived from tfrecord_path) is used for structuring output
                    # This was already correct.

                    sample_file_basename = f"{current_segment_name_from_frame}_frame{frame_idx:04d}_cam{camera_name_enum_val}"
                    image_path_relative = os.path.join(split_for_saving, sample_file_basename + ".jpg")
                    mask_path_relative = os.path.join(split_for_saving, sample_file_basename + ".png")

                    full_image_save_path = os.path.join(IMAGE_SAVE_DIR, image_path_relative)
                    full_mask_save_path = os.path.join(MASK_SAVE_DIR, mask_path_relative)
                    
                    # Ensure directories for the specific split are created (moved earlier)
                    # os.makedirs(os.path.dirname(full_image_save_path), exist_ok=True)
                    # os.makedirs(os.path.dirname(full_mask_save_path), exist_ok=True)

                    camera_image_pil.save(full_image_save_path)
                    Image.fromarray(new_mask_for_model).save(full_mask_save_path)

                    img_path_manifest = os.path.join("images", image_path_relative).replace("\\\\", "/")
                    mask_path_manifest = os.path.join("masks", mask_path_relative).replace("\\\\", "/")

                    manifest_data[split_for_saving].append({
                        "image": img_path_manifest,
                        "mask": mask_path_manifest
                    })
                    
                    current_config_samples_processed += 1 # Increment for this config attempt
                    global_sample_id_counter += 1       # Increment global total

                    if global_sample_id_counter % 100 == 0: # Report based on global counter
                        print(f"Processed {global_sample_id_counter} total samples so far...")
            # End of frame loop
        # End of tfrecord file loop for current config

        if current_config_samples_processed > 0:
            print(f"Successfully processed {current_config_samples_processed} samples using configuration: {path_config_entry['description']}")
            print("This configuration will be used. Halting further path searches.")
            processed_successfully = True
            break # Exit the loop over POTENTIAL_GCS_PATH_CONFIGS
        else:
            print(f"Configuration '{path_config_entry['description']}' did not yield any processable samples. Trying next configuration if available.")
    # End of loop over POTENTIAL_GCS_PATH_CONFIGS

    if not processed_successfully:
        print("\\n--- All GCS path configurations attempted. ---")
        print("Failed to process any samples. Please ensure:")
        print("1. At least one of the GCS paths in POTENTIAL_GCS_PATH_CONFIGS is correct.")
        print("2. The TFRecord files at that path contain 2D panoptic segmentation labels")
        print("   in 'frame.camera_labels[CAM_ID].panoptic_label' or ")
        print("   'frame.images[CAM_ID].camera_segmentation_label.panoptic_label'.")
        print("3. The user has appropriate permissions to access the GCS buckets/files.")

    print(f"Finished processing. Total unique samples processed across all successful configurations: {global_sample_id_counter}")
    
    # Save the manifest from the last successful attempt (or empty if all failed but one created manifest_data)
    with open(MANIFEST_FILE_PATH, 'w') as f:
        json.dump(manifest_data, f, indent=2)
    print(f"Manifest saved to {MANIFEST_FILE_PATH}")

if __name__ == '__main__':
    main() 