import re
import os
import data.timeseries_processing as tp
import numpy as np
import os
from patchify import unpatchify
import re
from tqdm import tqdm
import warnings
import inference.roots_segmentation as post
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*separator in column name.*")

def model_predict(image_path: str, patch_size: int, model) -> np.ndarray:
    """
    """
    # Extract ROI.
    image_roi, x, y, l, original_shape = tp.detect_extract(image_path)
    # Pad to be divisible by patch size.
    image_padded = tp.padder(image_roi, patch_size)

    # Calculate padding offsets
    hp = image_padded.shape[0] - image_roi.shape[0]
    wp = image_padded.shape[1] - image_roi.shape[1]
    top_p = hp // 2
    left_p = wp // 2

    # Patchify and predict.
    patches, shape = tp.patch_image(image_padded, patch_size)
    predictions = model.predict(patches / 255.0, verbose=0)
    predictions = predictions.reshape(shape[0], shape[1], patch_size, patch_size)
    predictions = unpatchify(predictions, image_padded.shape)
    predictions = (predictions > 0.5).astype(np.uint8)
    # Noise removal.
    predictions = tp.remove_noise(predictions, min_area=50)
    predictions_roi = predictions[top_p:top_p+l, left_p:left_p+l]
    # Revert ROI extraction.
    full_mask = np.zeros(original_shape, dtype=np.uint8)
    full_mask[y:y+l, x:x+l] = predictions_roi

    return full_mask


def extract_all_numbers(filename: str) -> tuple:
    """
    Extracts all numeric groups from a filename and returns them as a tuple
    for sorting. Falls back to (0,) if no numbers found.
    """
    numbers = re.findall(r'\d+', filename)
    return tuple(int(n) for n in numbers) if numbers else (0,)


def predict_timeseries(folder_path: str, patch_size: int, root_model, shoot_model):
    """
    """
    image_files = sorted(
        [f for f in os.listdir(folder_path) if f.lower().endswith((".png", ".jpg", ".jxl"))],
        key=extract_all_numbers
    )
    # Define expected_centers.
    expected_centers = [
        (1000, 550, 1), (1500, 550, 2), (2000, 550, 3),
        (2500, 550, 4), (3000, 550, 5)
    ]

    # Initialise results dict.
    results = {}

    if not image_files:
        print(f"No valid images are found in {folder_path}")
        return results
    start_coords_per_plant = {i: None for i in range(5)}
    MIN_RELIABLE_T = 4
    for t, filename in enumerate(tqdm(image_files, desc="Predicting mask")):
        image_path = os.path.join(folder_path, filename)
        results[filename] = {}
        try:
            root_mask = model_predict(
                image_path,
                patch_size,
                root_model
            )
            shoot_mask = model_predict(
                image_path,
                patch_size,
                shoot_model
            )
            # Save the predictions in the dictionaries.
            results[filename]["roots"] = root_mask
            results[filename]["shoots"] = shoot_mask

            # Segment and select primary/lateral roots.
            branch_df, skeleton_ob, plant_bboxes = post.segment_roots(
                root_mask, expected_centers,
                reconnect_max_dist=40.0,
                known_start_coords=start_coords_per_plant if t >= MIN_RELIABLE_T else None
            )
            if branch_df is not None:
                for plant_idx in range(5):
                    primary = branch_df[
                        (branch_df["plant"] == plant_idx) &
                        (branch_df["root_type"] == "Primary")
                    ]
                    if not primary.empty:
                        # Store the topmost primary coord as the known start
                        # top_row = primary.loc[primary["image-coord-src-0"].idxmin()]
                        # start_coords_per_plant[plant_idx] = (
                        #     top_row["image-coord-src-0"],
                        #     top_row["image-coord-src-1"]
                        # )
                        top_row = primary.loc[primary["image-coord-src-0"].idxmin()]
                        new_coord = (
                            top_row["image-coord-src-0"],
                            top_row["image-coord-src-1"]
                        )
                        current = start_coords_per_plant[plant_idx]
                        if current is None or new_coord[0] < current[0]:
                            start_coords_per_plant[plant_idx] = new_coord

            # branch_df, skeleton_ob, plant_bboxes = post.segment_roots(root_mask, expected_centers, reconnect_max_dist=40.0)
            primary_mask, lateral_mask = post.get_masks(branch_df, skeleton_ob, root_mask.shape)
            results[filename]["lateral"] = lateral_mask
            results[filename]["primary"] = primary_mask
            results[filename]["bboxes"] = plant_bboxes
            results[filename]["branches"] = branch_df
            results[filename]["skeleton"] = skeleton_ob

        except Exception as e:
            print(f"Failed to predict for {filename}: {e}")
            results[filename]["roots"] = None
            results[filename]["shoots"] = None
            results[filename]["lateral"] = None
            results[filename]["primary"] = None
            results[filename]["bboxes"] = None
            results[filename]["branches"] = None
            results[filename]["skeleton"] = None
    return results
