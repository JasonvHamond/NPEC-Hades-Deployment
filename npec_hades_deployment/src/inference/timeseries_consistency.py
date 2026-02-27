import sys
import os
import re
sys.path.append(os.path.abspath(".."))
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import networkx as nx
import pandas as pd
import skimage
import cv2
import math
import typer

from data import timeseries_processing as tp
import inference.roots_segmentation as post

from inference.timeseries_correction import (
    path_similarity,
    tip_displacement,
    find_consistent_growth_path,
    interpolate_path,
    path_to_mask
)

def get_plant_mask(predictions, filename, plant_idx, shape):
    """
    Get the binary mask for a plant by combining branch and skeleton predictions.

    Parameters:
    - predictions (dict):
        Dictionary containing model predictions for each filename. Each entry should have:
        - "branches": DataFrame with branch information, including "plant" and "root_type" columns.
        - "skeleton": 2D array representing the skeleton prediction for the image.
    - filename (str):
        The filename for which to extract the plant mask.
    - plant_idx (int):
        The index of the plant for which to extract the mask.
    - shape (tuple):
        The desired shape of the output mask (height, width).
    
    Returns:
    - mask (2D array):
        A binary mask of the specified shape.
    """
    branch_df = predictions[filename]["branches"]
    skeleton_ob = predictions[filename]["skeleton"]
    mask, _ = post.get_masks(branch_df, skeleton_ob, shape, plant_idx=plant_idx)
    return mask


def flag_inconsistencies(
    predictions,
    coords_per_plant,
    # folder_path,
    min_similarity=0.7,
    correct=True
):
    """
    Check temporal consistency of root paths and flag plants with inconsitencies, big drops in length,
    or a significantly lower start node. Optionally, try to correct flagged plants by finding a more consistent path.

    Parameters:
    - predictions (dict):
        Dictionary containing model predictions for each filename. Each entry should have:
        - "branches": DataFrame with branch information, including "plant" and "root_type" columns.
        - "primary": 2D array representing the primary root mask prediction for the image.
        - "skeleton": 2D array representing the skeleton prediction for the image.
    - coords_per_plant (dict):
        Dictionary mapping plant indices to lists of coordinate arrays for each timeframe.
    - folder_path (str):
        Path to the folder containing the original images, used for loading and comparison.
    - min_similarity (float):
        Minimum path similarity threshold to flag inconsistencies. Should be between 0 and 1.
    - correct (bool):
        Whether to attempt correction of flagged plants by finding a more consistent growth path.

    Returns:
    - predictions_new (dict):
        Updated predictions dictionary with flagged plants corrected if correction is enabled.
    """
    
    filenames = list(predictions.keys())
    predictions_new = predictions.copy()

    for plant_idx, coords_list in coords_per_plant.items():
        filenames = list(predictions.keys())
        print(f"Checking consistency for Plant {plant_idx} across {len(coords_list)} timeframes.")
        for t in range(1, len(coords_list)):
            # Calculate similarity and displacement between current and previous coordinates.
            sim = path_similarity(coords_list[t-1], coords_list[t])
            disp = tip_displacement(coords_list[t-1], coords_list[t])
            curr_coords = coords_per_plant[plant_idx][t]
            prev_coords = coords_per_plant[plant_idx][t-1]
            # Check for big drops in the length of the root compared to previous timeframe.
            prev_primary_length = len(prev_coords)
            curr_primary_length = len(curr_coords) if len(curr_coords) > 0 else 0
            length_dropped = prev_primary_length - curr_primary_length > 10
            prev_start_y = prev_coords[:, 0].min() if len(prev_coords) > 0 else None
            curr_start_y = curr_coords[:, 0].min() if len(curr_coords) > 0 else None
            # Also check if the start node is lower than previous timeframe.
            start_dropped = (
                prev_start_y is not None and 
                curr_start_y is not None and 
                curr_start_y - prev_start_y > 100
            )
            curr_coords = coords_per_plant[plant_idx][t]
            has_next = t + 1 < len(coords_per_plant[plant_idx])
            next_coords = coords_per_plant[plant_idx][t+1] if has_next else np.array([])
            filename_curr = filenames[t]

            # Flag the image if similarity is lower than threshold,
            # length of the current root dropped significantly, or start node is lower.
            if (sim < min_similarity or length_dropped or start_dropped or
                (len(curr_coords) == 0 and len(prev_coords) > 0 and len(next_coords) > 0)):
                print(f"Flagged Plant {plant_idx}, t{t-1}>t{t}")
                print(f"similarity={sim:.2f}, displacement={disp:.1f}px. {filename_curr}")
                branch_df = predictions_new[filename_curr]["branches"]
                has_primary = len(branch_df[
                    (branch_df["plant"] == plant_idx) &
                    (branch_df["root_type"] == "Primary")
                ]) > 0
                has_laterals = len(branch_df[
                    (branch_df["plant"] == plant_idx) &
                    (branch_df["root_type"] == "Lateral")
                ]) > 0
                if not has_primary and not has_laterals:
                    if len(prev_coords) > 0 and len(next_coords) > 0:
                        print(f"Plant {plant_idx} t{t}: No primary or lateral, interpolating.")
                        coords_interp = interpolate_path(prev_coords, next_coords)
                        mask_interp = path_to_mask(coords_interp, predictions_new[filename_curr]["primary"].shape)
                        coords_per_plant[plant_idx][t] = coords_interp
                        predictions_new[filename_curr]["primary"] = mask_interp
                    else:
                        print(f"Plant {plant_idx} t{t}: No primary, lateral, or neighbor frames, skipping.")
                    continue

                # load original images for comparison.
                # original_curr = tp.load_image(os.path.join(folder_path, filename_curr))

                # Use union of both coords for crop so both frames show the same region.
                coords_curr = coords_list[t]
                coords_prev = coords_list[t-1]
                valid_coords = [c for c in [coords_curr, coords_prev] if len(c) > 0]

                if len(valid_coords) == 0:
                    print("No coords available for crop, skipping.")
                    continue

                # all_coords = np.concatenate(valid_coords)

                # ymin, xmin = all_coords.min(axis=0).astype(int)
                # ymax, xmax = all_coords.max(axis=0).astype(int)

                # pad = 100
                # h, w = original_curr.shape[:2]
                # ymin, ymax = max(0, ymin - pad), min(h, ymax + pad)
                # xmin, xmax = max(0, xmin - pad), min(w, xmax + pad)

                # if correction is enabled, try to find a more consistent path and plot the results.
                if correct:
                    print(f"Attempting correction for Plant {plant_idx}, t{t-1}>t{t}...")
                    predictions_new, _ = find_consistent_growth_path(
                        predictions_new,
                        filenames,
                        coords_per_plant,
                        plant_idx,
                        t
                    )
                    
    return predictions_new