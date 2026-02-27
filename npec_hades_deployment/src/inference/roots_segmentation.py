#
from tkinter import Image
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import os
import re
import warnings
from typing import List, Tuple, Union, Optional
import cv2
import networkx as nx
import pandas as pd
import skan
from skan import Skeleton, summarize
from rich.progress import track
import skimage
from skimage.draw import line
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
from scipy.spatial import KDTree
from itertools import combinations
import networkx
from typing import Dict
from skimage.morphology import disk
from skimage.util import img_as_bool
from scipy.ndimage import convolve
from PIL import Image
import pillow_jxl

import matplotlib
matplotlib.use('TkAgg')
#sys.path.append("C:/Users/phyto/Documents/NPEC Arabidopsis Hades Pipeline")  # Path to the folder containing pyphenotyper
import inference.rsml as rsml
# from inference.timeseries_consistency import flag_inconsistencies, get_plant_mask

expected_centers = [
    (1000, 550, 1),
    (1500, 550, 2),
    (2000, 550, 3),
    (2500, 550, 4),
    (3000, 550, 5)]

def load_in_mask(path: str) -> Optional[np.ndarray]:
    """
    This function takes in a path to the mask, loads it in, segments it, removes all the object that are smaller than
    an average area of all segmented instances, transfers the image back to BGR

    Author: Fedya Chursin, 220904@buas.nl

    :param path: Path to the mask image.
    :type path: str
    :return: Loaded in mask image with root segmentation in BGR format.
    :rtype: np.ndarray
    """
    # Check if the path is a string
    if not isinstance(path, str):
        warnings.warn(
            f"Expected 'path' to be a string, but got {type(path).__name__}.")
        return None
    # Read the mask
    mask = cv2.imread(path, 0)
    if mask is None:
        raise ValueError(f"Could not read the image at {path}")
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return mask

def draw_root(coordinates: np.array, image: np.array, x_offset: float, y_offset: float, colour: list = (0, 0, 0), thickness: int = 1) -> np.array:
    """
    This function visualises the given coordinates on the image with the provided colour and thickness.

    Author: Wesley van Gaalen, 224682@buas.nl

    :param coordinates: array of (y, x) coordinates that has to be visualised
    :param image: image for visualisation
    :param x_offset: left bound for the current plant - used to draw on the real image using the coordinates from the sliced image that focuses on a certain plant
    :param y_offset: top bound for the current plant - used to draw on the real image using the coordinates from the sliced image that focuses on a certain plant
    :param colour: colour with which to draw
    :param thickness: thickness of the line, with 1 being the default single pixel
    :return: returns image where the given coordinates are highlighted with the provided colour
    """
    # loop through all the coordinates pairs and visualise them
    for y, x in coordinates:
        y += y_offset
        x += x_offset
        # Make the lines a bit thicker so that they're more visible
        for i in range(-thickness, thickness + 1):
            for j in range(-thickness, thickness + 1):
                if 0 <= y + i < image.shape[0] and 0 <= x + j < image.shape[1]:
                    image[y + i, x + j] = colour
    return image

def get_closest_cord(subset_branch: pd.DataFrame, row: pd.DataFrame) -> pd.DataFrame:
    """
    Things function finds the closest branch to the given row, and returns the branch that has the closest angle to the given row.
    
    Author: Wesley van Gaalen, 224682@buas.nl

    :param subset_branch: dataframe with all the branches.
    :param row: specific row of the dataframe with the branch information.
    :return: new_row: row of the dataframe with the branch information that has the closest angle to the given row. If no row is close to it, it returns an empty dataframe.
    """
    # Get the coordinates of the point
    point_x = row["image-coord-dst-1"]
    point_y = row["image-coord-dst-0"]
    # Get the angle of the row
    angle = row["angle"]
    # Set the distance for the closest branches
    threshold_distance = 45
    # Calculate the distance between the source point and all the other points
    src_distance = np.sqrt(
        (subset_branch["coord-src-1"] - point_x) ** 2
        + (subset_branch["coord-src-0"] - point_y) ** 2
    )
    # Filter the dataframe to the branches that are close to the source point, but not the same branch, and not the primary root.
    filtered_df = subset_branch[
        (
                ((src_distance <= threshold_distance))
                & ~(subset_branch["node-id-src"] == row["node-id-src"])
                & ~(subset_branch["node-id-dst"] == row["node-id-dst"])
        )
        & (subset_branch["root_type"] != "Primary")
        ]
    # If there are branches that are close to the source point, find the one with the closest angle to the given row.
    if not filtered_df.empty:
        new_row = filtered_df.iloc[(
                                           filtered_df["angle"] - angle).abs().argsort()[:1]]
        return new_row
    else:
        return pd.DataFrame()

def get_x_y(row: pd.DataFrame) -> Tuple[float, float, float, float]:
    """
    This function returns the start and destinations coordinates of a branch.

    Author: Wesley van Gaalen, 224682@buas.nl
    
    :param row: row of the dataframe with the branch information.
    :return: x_src: x coordinate of the source point.
    :return: y_src: y coordinate of the source point.
    :return: x_dst: x coordinate of the destination point.
    :return: y_dst: y coordinate of the destination point.
    """
    x_src = row["image-coord-src-0"]
    y_src = row["image-coord-src-1"]
    x_dst = row["image-coord-dst-0"]
    y_dst = row["image-coord-dst-1"]
    return x_src, y_src, x_dst, y_dst

def calculate_angle(row: pd.DataFrame) -> float:
    """
    This function calculates the angle of a branch in degrees.

    Author: Wesley van Gaalen, 224682@buas.nl

    :param row: row of the dataframe with the branch information.
    :return: angle_degrees: angle of the branch in degrees.
    """
    # Get the coordinates of the branch
    x1, y1, x2, y2 = get_x_y(row)
    # Calculate the angle of the branch
    delta_x = x2 - x1
    delta_y = y2 - y1
    # Calculate the angle in radians
    angle_radians = math.atan2(delta_y, delta_x)
    # Convert the angle to degrees
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees

def follow_lateral_path(subset_branch):
    """

    Author: Wesley van Gaalen, 224682@buas.nl

    This function checks for each branch from the primary root, and follows the path of the lateral root.
    :param subset_branch: subset of the dataframe with all the branches. It should have an additional 'plant' and 'root_type' column, with primary roots marked as 'Primary' and the rest as 'None'.
    :return: subset_branch: dataframe with the roots assigned to the plants.
    """
    subset_branch_plant = subset_branch[subset_branch["plant"] != "None"]
    for plant in sorted(subset_branch_plant["plant"].unique()):
        # We convert the plant to an integer if it is not NaN
        if plant == "None" or np.isnan(plant):
            continue
        plant = int(plant)
        # Filter the dataframe to the current plant and only the primary roots
        filtered_df = subset_branch[subset_branch["plant"] == plant]
        filtered_df = filtered_df[filtered_df["root_type"] == "Primary"]

        root_id = 0

        # Iterate over the rows where the plant is the current plant and the root type is primary
        for index, row in filtered_df.iterrows():
            following_path = True
            # Reset the steps taken
            steps_taken = 0
            # Start from the primary root and follow the root path to the end.
            while following_path:
                # If it's the first step, so comming from the primary root, find the closest row to the primary root.
                if steps_taken == 0:
                    # Find the closest row to the primary root
                    closest_row = get_closest_cord(subset_branch, row)
                elif steps_taken > 10:
                    following_path = False
                    continue
                else:
                    # Find the closest row to the previous row
                    closest_row = get_closest_cord(subset_branch, closest_row)
                # Check if no closest row found, then stop following the path
                if closest_row.empty:
                    following_path = False
                    continue
                # Update the index and row with the closest one found
                index = closest_row.index[0]
                # Find the complete row in the original dataframe
                closest_row = subset_branch.loc[index]
                # Assign the plant to the closest row
                subset_branch.loc[index, "plant"] = plant
                subset_branch.loc[index, "root_type"] = "Lateral"
                subset_branch.loc[index, "root_id"] = f"Lateral_{root_id}"
                # Increment the steps taken
                steps_taken += 1
            root_id += 1
    return subset_branch

def draw_lateral(subset_branch: pd.DataFrame, subset_skeleton_ob_arr: Skeleton, img2: np.array,
                 colors: list = [(0, 128, 255), (240, 32, 160), (255, 0, 0), (255, 0, 255), (0, 255, 0)]) -> np.array:
    """
    This function draws the lateral roots on the image. It uses the subset_dataframe and draaws each cordinate of the lateral branches.

    Author: Wesley van Gaalen, 224682@buas.nl
    
    :param subset_branch: subset of the dataframe with all the branches. It should have an additional 'plant' and 'root_type' column, with primary roots marked as 'Primary' and the rest as 'None'.
    :param subset_skeleton_ob_arr: skeleton object array of the image.
    :param img2: image off the petri dish to draw the roots on.
    :param colors: list of colors to draw the lateral roots with.
    :return: img2: image with the roots drawn on it.
    """
    # We look one more time over the dataframe, and draw all of the lateral roots on the image here.
    for index, row in subset_branch.iterrows():
        if row.get("root_type") == "Primary" or pd.isna(row.get("plant")):
            continue
        try:
            color_index = int(row["plant"])
            xy = subset_skeleton_ob_arr.path_coordinates(index)
            img2 = draw_root(xy, img2, 0, 0, (colors[color_index]))
        except (ValueError, TypeError, IndexError):
            continue  # skip invalid values safely
    return img2


def apply_conversion_factor(number: float) -> float:
    """
    Apply the pixel to real-life conversion factor to a given number.

    Author: Wesley van Gaalen, 224682@buas.nl
    
    :param number: The number to be converted.
    :return: The converted number.
    :example:
    >>> apply_conversion_factor(412)
    22.350813743218804
    Vince here: scale conversion from 18.43 px/mm to 24.28 px/mm fixed on March 14, 2025 - Petri dish from 150 mm to 113.8797 mm) 
    """

    if not isinstance(number, (int, float)):
        raise TypeError("The input must be an integer or a float.")

    # The petri dish plate size in mm
    plate_size_mm = 113.8797
    # The petri dish plate size in pixels
    plate_size_pixels = 2765
    # The conversion factor
    conversion_factor = plate_size_mm / plate_size_pixels
    return number * conversion_factor

def get_primary_landmarks(landmark_df):
    """
    Get the primary landmarks of the plant. These are the root top (starting of)
    and the root tip (ending of) the primary root. This function finds them by
    looking for the highest location of the primary root, and assumes that it's
    the top of the primary root. The lowest location of the primary root is
    assumed to be the tip of the primary root.

    :param landmark_df: The dataframe with the root information.
    :return: landmark_df_copy: The dataframe with the primary landmarks marked.
    """

    if not isinstance(landmark_df, pd.DataFrame):
        raise TypeError("The input must be a pandas DataFrame.")

    if len(landmark_df) == 0:
        return landmark_df

    # Make a copy of the dataframe because of SettingWithCopyWarning from pandas
    landmark_df_copy = landmark_df.copy()

    x =0
    for index, row in landmark_df.iterrows():
        if row["root_type"] == "Primary":
            landmark_df_copy.loc[index, "landmark"] = "Primary_Junction"
            landmark_df_copy.loc[index, "root_id"] = f"Junction_{x}"
            x += 1

    # Create subset with only primary root branches.
    primary_subset = landmark_df[landmark_df["root_type"] == "Primary"]

    if primary_subset.empty:
        print(f"Warning: no primary root found in plant subset, skipping primary landmarks.")
        return landmark_df_copy


    if len(primary_subset) == 1:
        new_df = primary_subset.copy()
        landmark_df_copy = pd.concat([landmark_df_copy, new_df], ignore_index=True)
        primary_subset = pd.concat([primary_subset, new_df], ignore_index=True)
        primary_root_top_row = primary_subset.iloc[0]
        primary_root_tip_row = primary_subset.iloc[1]
    else:

        # Find the primary root top by looking at the lowest node-id-src, which corresponds to the highest location of the primary root.
        primary_root_top_row = primary_subset.loc[primary_subset["node-id-src"].idxmin()]

        # Find the primary root tip by looking at the highest node-id-dst, which corresponds to the lowest location of the primary root.
        primary_root_tip_row = primary_subset.loc[primary_subset["node-id-dst"].idxmax()]

    # Change the landmark in the dataframe to the primary root tip and top.
    landmark_df_copy.loc[primary_root_tip_row.name, "landmark"] = "Primary_root_tip"
    landmark_df_copy.loc[primary_root_top_row.name, "landmark"] = "Primary_root_top"
    landmark_df_copy.loc[primary_root_tip_row.name, "root_id"] = "Primary"
    landmark_df_copy.loc[primary_root_top_row.name, "root_id"] = "Primary"


    # cv2.circle(img, (primary_root_tip_row["image-coord-dst-1"], primary_root_tip_row["image-coord-dst-0"]), 10, (0, 0, 255), -1)
    # cv2.circle(img, (primary_root_top_row["image-coord-src-1"], primary_root_top_row["image-coord-src-0"]), 10, (255, 0, 255), -1)

    return landmark_df_copy

def get_lateral_landmarks(landmark_df):
    """
    Get the lateral landmarks of the plant. These are the ending of all the lateral roots.
    This function finds them by looking if the ending of each branch is not the starting of any other branch.
    Since sometimes lateral roots are not connected, it can give false landmarks. That's why, when the lateral root is not connected to the primary root, it's excluded.

    :param landmark_df: The dataframe with the root information.
    :return: landmark_df_copy: The dataframe with the lateral landmarks marked.
    """

    if not isinstance(landmark_df, pd.DataFrame):
        raise TypeError("The input must be a pandas DataFrame.")

    if len(landmark_df) == 0:
        return landmark_df

    # Make a copy of the dataframe because of SettingWithCopyWarning from pandas
    landmark_df_copy = landmark_df.copy()
    # Create subset with only lateral root branches.
    lateral_subset = landmark_df[landmark_df["root_type"] == "Lateral"]
    
    primary_rows = landmark_df[landmark_df["root_type"] == "Primary"]
    if primary_rows.empty:
        print("Warning: no primary root found, skipping lateral landmark detection.")
        return landmark_df_copy
    
    # Get the primary skeleton id
    primary_skeleton_id = landmark_df[landmark_df["root_type"] == "Primary"]["skeleton-id"].iloc[0]

    # Loop through each row in the lateral subset, and check if the ending of the branch is not the starting of any other branch.
    # Also checking if the lateral root is connected to the primary root.
    for index, row in lateral_subset.iterrows():
        if not landmark_df["node-id-src"].isin([row["node-id-dst"]]).any() and \
                row["skeleton-id"] == primary_skeleton_id:
            landmark_df_copy.loc[index, "landmark"] = "Lateral_root_tip"
            # cv2.circle(img, (row["image-coord-dst-1"], row["image-coord-dst-0"]), 10, (0, 255, 0), -1)

    return landmark_df_copy

def process_landmark_df(landmark_df):
    """
    This function processes the landmark dataframe to clean it up and only get the important things.
    It removes all "None", and renames the columns to "x" and "y".
    It also sorts the dataframe by the landmark name, and resets the index.

    :param landmark_df: The dataframe with the root information.
    :return: landmark_df_processed: The processed dataframe with the landmarks.
    """
    if not isinstance(landmark_df, pd.DataFrame):
        raise TypeError("The input must be a pandas DataFrame.")
    if len(landmark_df) == 0:
        return landmark_df
    # Remove all "None" branches, since these are not of value.
    landmark_df = landmark_df[landmark_df["landmark"] != "None"]
    # Drop all rows with NaN values
    landmark_df = landmark_df.dropna()

    if len(landmark_df) == 0:
        return landmark_df
    # Remove the primary root top, because for this one we need source points. We add it later.
    landmark_df_processed = landmark_df[landmark_df["landmark"] != "Primary_root_top"]

    # Only keep relevant columns
    landmark_df_processed = landmark_df_processed[
        ["landmark", "image-coord-dst-0", "image-coord-dst-1", "root_id", "image-coord-src-0"]
    ]
    # Rename columns for more clarity
    landmark_df_processed = landmark_df_processed.rename(
        columns={"image-coord-dst-0": "y", "image-coord-dst-1": "x", "image-coord-src-0": "y_src"}
    )

    # Add the primary root top back to the dataframe
    new_row = landmark_df[landmark_df["landmark"] == "Primary_root_top"]
    new_row = new_row[["landmark", "image-coord-src-0", "image-coord-src-1", "root_id", "node-id-src"]]
    new_row = new_row.rename(
        columns={"image-coord-src-0": "y", "image-coord-src-1": "x", "node-id-src": "y_src"}
    )

    landmark_df_processed.loc[len(landmark_df_processed) + 1] = new_row.iloc[0]

    # Sort the dataframe by the landmark name
    landmark_df_processed = landmark_df_processed.sort_values(by=["root_id"])
    # Reset the index
    landmark_df_processed = landmark_df_processed.reset_index(drop=True)

    return landmark_df_processed

def get_landmarks(subset_plant):
    """
    This is the main function for getting the landmarks of a root structure.
    It first gets the primary landmarks, and then the lateral landmarks.
    After both, it cleans up the dataframe.

    :param subset_plant: The dataframe with the root information.
    :return: landmark_df: The processed dataframe with the landmarks.
    """
    empty_df = pd.DataFrame(columns=["landmark", "y", "x", "root_id", "y_src", "plant"])
    # Create a copy of the dataframe to avoid SettingWithCopyWarning from pandas
    landmark_df = subset_plant.copy()
    # Add a column for the landmarks
    landmark_df["landmark"] = "None"
    if not (landmark_df["root_type"] == "Primary").any():
        print("Warning: no primary root in plant, skipping landmark detection.")
        return empty_df
    # Get the primary landmarks
    landmark_df = get_primary_landmarks(landmark_df)

    # Get the lateral landmarks
    landmark_df = get_lateral_landmarks(landmark_df)

    # Process the landmark dataframe
    landmark_df = process_landmark_df(landmark_df)
    if landmark_df.empty or "root_id" not in landmark_df.columns:
        print("Warning: landmark processing returned empty result.")
        return empty_df


    return landmark_df

def draw_landmarks(landmarked_img, landmark_df):
    """
    This function draws the landmarks on the image. It does this by drawing a circle at the x and y coordinates of the landmarks.

    Author: Wesley van Gaalen, 224682@buas.nl

    :param landmarked_img: The image of the petri dish.
    :param landmark_df: The dataframe with the landmarks.
    :return: landmarked_img: The image with the landmarks drawn on it.
    """
    if landmark_df.empty or "y_src" not in landmark_df.columns:
        print("Warning: no landmarks to draw, skipping.")
        return landmarked_img
    # Loop through the rows of the dataframe
    root_id = 0
    primary_junction_root_id = 0
    landmark_df = landmark_df.sort_values(by=["y_src"])
    for index, row in landmark_df.iterrows():
        if row["landmark"] == "Primary_root_tip":
            color = (255, 0, 255)
            landmark_df.loc[index, "root_id"] = "Primary Tip"
            row["root_id"] = "Primary Tip"
        elif row["landmark"] == "Primary_root_top":
            color = (255, 0, 0)
            landmark_df.loc[index, "root_id"] = "Primary Top"
            row["root_id"] = "Primary Top"
        elif row["landmark"] == "Lateral_root_tip":
            color = (0, 255, 0)
            landmark_df.loc[index, "root_id"] = f"L{root_id}"
            row["root_id"] = f"L{root_id}"
            root_id += 1
        elif row["landmark"] == "Primary_Junction":
            color = (0, 0, 255)
            landmark_df.loc[index, "root_id"] = f"J{primary_junction_root_id}"
            row["root_id"] = f"J{primary_junction_root_id}"
            primary_junction_root_id += 1

        # Get the x and y coordinates of the landmark
        x = int(row["x"])
        y = int(row["y"])
        # Draw a circle at the x and y coordinates
        cv2.circle(landmarked_img, (x, y), 10, color, -1)
        landmarked_img = cv2.putText(landmarked_img, row["root_id"], (x, (y - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                                     1, (0, 0, 0), 2, cv2.LINE_AA)
    return landmarked_img

def merge_shoot_root(shoot_mask, root_mask):
    """
    This function adds two masks together. The masks have to be the same size, and only have 2 unique values: 0 and 255.

    Author: Wesley van Gaalen, 224682@buas.nl

    :param shoot_mask: The mask of the shoot.
    :param root_mask: The mask of the roots.
    """
    # Make a copy of the root mask
    shoot_root_mask = root_mask.copy()
    # Add the shoot mask to the root mask
    shoot_root_mask[shoot_mask == 255] = 255
    return shoot_root_mask

def add_shoot_image_mask(image, shoot_mask):
    """
    This function adds the shoot mask to the image. It does this by setting the pixels of the shoot mask to a specific color in the image.

    Author: Wesley van Gaalen, 224682@buas.nl

    :param image: The image of the petri dish.
    :param shoot_mask: The mask of the shoot.
    :return: image: The image with the shoot mask added to it.
    """
    image[shoot_mask == 1] = [10, 88, 32]
    return image

def reconnect_skeleton(skeleton, max_dist=20):
    """
    Reconnects disconnected skeleton components with only one connection per region,
    favoring the closest and most downward (larger Y) partner.
    """
    selem = skimage.morphology.square(1)
    expand_skel = skimage.morphology.dilation(skeleton, selem)

    labeled = skimage.measure.label(expand_skel)
    props = skimage.measure.regionprops(labeled)

    connections = []

    for i, region1 in enumerate(props):
        best_score = float('inf')
        best_coords = None

        coords1 = region1.coords
        y1_mean = np.mean(coords1[:, 0])

        for j, region2 in enumerate(props):
            if i == j:
                continue

            coords2 = region2.coords
            y2_mean = np.mean(coords2[:, 0])

            # Favor downward connections: region2 must be lower than region1
            if y2_mean < y1_mean:
                continue

            dists = np.sqrt(((coords1[:, None] - coords2) ** 2).sum(-1))
            min_dist = np.min(dists)

            if min_dist < max_dist and min_dist < best_score:
                idx1, idx2 = np.unravel_index(np.argmin(dists), dists.shape)
                best_score = min_dist
                best_coords = (coords1[idx1], coords2[idx2])

        # Connect to the best match if found
        if best_coords:
            (y0, x0), (y1, x1) = best_coords
            rr, cc = line(y0, x0, y1, x1)
            skeleton[rr, cc] = 1
            connections.append((i, (y0, x0), (y1, x1)))

    return skeleton

def grow_bounding_boxes(
    binary_image,
    expected_centers,
    expansion_step=5,
    initial_box_halfsize_x=200,
    initial_box_halfsize_y=200,
    max_empty_expansions=200,
    stop_on_overlap=True
):
    """
    Grows bounding boxes from seed centers with:
    - Asymmetric initial box sizes (x/y)
    - First iteration refits the box tightly around actual root pixels
    - Direction-specific expansion control (only reset the direction that grew)
    - Optional: stop expansion in a direction when overlapping with another box

    Returns:
        - original_boxes: boxes from center and halfsize (pre-root check)
        - initial_boxes: boxes refitted to actual root pixels
        - final_boxes: result after full expansion
    """

    # Dilate to enhance weak/disconnected roots
    kernel = np.ones((5, 5), np.uint8)
    binary_image = cv2.dilate(binary_image, kernel, iterations=1)
    binary_image = (binary_image > 0).astype(np.uint8)

    height, width = binary_image.shape
    num_plants = len(expected_centers)

    original_boxes = []
    initial_boxes = []
    final_boxes = []
    empty_counters = []
    can_expand = []

    for cx, cy, _ in expected_centers:
        ymin_raw = max(cy - initial_box_halfsize_y, 0)
        ymax_raw = min(cy + initial_box_halfsize_y, height)
        xmin_raw = max(cx - initial_box_halfsize_x, 0)
        xmax_raw = min(cx + initial_box_halfsize_x, width)
        original_boxes.append([ymin_raw, ymax_raw, xmin_raw, xmax_raw])

        region = binary_image[ymin_raw:ymax_raw, xmin_raw:xmax_raw]
        ys, xs = np.where(region > 0)

        ymin, ymax, xmin, xmax = ymin_raw, ymax_raw, xmin_raw, xmax_raw
        if len(xs) > 0 and len(ys) > 0:
            ymin = max(ymin_raw + int(ys.min()) - 1, 0)
            ymax = min(ymin_raw + int(ys.max()) + 2, height)
            xmin = max(xmin_raw + int(xs.min()) - 1, 0)
            xmax = min(xmin_raw + int(xs.max()) + 2, width)

        initial_boxes.append([ymin, ymax, xmin, xmax])
        final_boxes.append([ymin, ymax, xmin, xmax])
        empty_counters.append({"up": 0, "down": 0, "left": 0, "right": 0})
        can_expand.append({"up": True, "down": True, "left": True, "right": True})

    def overlaps_any(proposed_box, current_idx):
        pymin, pymax, pxmin, pxmax = proposed_box
        for j, (oymin, oymax, oxmin, oxmax) in enumerate(final_boxes):
            if j == current_idx:
                continue
            if not (pxmax <= oxmin or pxmin >= oxmax or pymax <= oymin or pymin >= oymax):
                return True
        return False

    still_expanding = True
    while still_expanding:
        still_expanding = False

        for i in range(num_plants):
            ymin, ymax, xmin, xmax = final_boxes[i]

            for direction in ["up", "down", "left", "right"]:
                if not can_expand[i][direction]:
                    continue

                if direction == "up":
                    new_ymin = max(ymin - expansion_step, 0)
                    region = binary_image[new_ymin:ymin, xmin:xmax]
                    proposed = [new_ymin, ymax, xmin, xmax]
                elif direction == "down":
                    new_ymax = min(ymax + expansion_step, height)
                    region = binary_image[ymax:new_ymax, xmin:xmax]
                    proposed = [ymin, new_ymax, xmin, xmax]
                elif direction == "left":
                    new_xmin = max(xmin - expansion_step, 0)
                    region = binary_image[ymin:ymax, new_xmin:xmin]
                    proposed = [ymin, ymax, new_xmin, xmax]
                elif direction == "right":
                    new_xmax = min(xmax + expansion_step, width)
                    region = binary_image[ymin:ymax, xmax:new_xmax]
                    proposed = [ymin, ymax, xmin, new_xmax]
                else:
                    continue

                pixel_count = np.sum(region)

                if stop_on_overlap and overlaps_any(proposed, i):
                    can_expand[i][direction] = False
                    continue

                if pixel_count > 0:
                    # Apply new box bounds
                    if direction == "up":
                        final_boxes[i][0] = proposed[0]
                    elif direction == "down":
                        final_boxes[i][1] = proposed[1]
                    elif direction == "left":
                        final_boxes[i][2] = proposed[2]
                    elif direction == "right":
                        final_boxes[i][3] = proposed[3]

                    # Only reset the direction that grew
                    can_expand[i][direction] = True
                    empty_counters[i][direction] = 0
                    still_expanding = True
                else:
                    empty_counters[i][direction] += 1
                    if empty_counters[i][direction] >= max_empty_expansions:
                        can_expand[i][direction] = False

    return original_boxes, initial_boxes, final_boxes

def detect_roots(mask: str, expected_centers = expected_centers ) -> tuple:
    """
    Detect plants by expanding bounding boxes from seed positions using a binary root mask.
    Also returns the root mask image with original and expanded bounding boxes overlaid.

    :param root_mask_path: Path to binary root mask image (white = root).
    :param seed_sk_path: (Unused, kept for compatibility)
    :return: (List of bounding boxes, root mask with overlays)
    """

    # Load binary root mask
    #root_mask = cv2.imread(root_mask_path, cv2.IMREAD_GRAYSCALE)
    #root_mask = ((root_mask > 0) * 1).astype(np.uint8)

    root_mask = mask

    # Define fixed expected centers (x, y, id)
    initial_box_halfsize_x = 200
    initial_box_halfsize_y = 200

    # Expand bounding boxes
    original_boxes, initial_boxes, final_boxes = grow_bounding_boxes(
        binary_image=root_mask,
        expected_centers=expected_centers,
        expansion_step=1,
        initial_box_halfsize_x=initial_box_halfsize_x,
        initial_box_halfsize_y=initial_box_halfsize_y,
        max_empty_expansions=100,
        stop_on_overlap=True)

    return final_boxes

def prepare_data_for_segmentation(path: str, expected_centers=expected_centers) -> List[np.ndarray]:
    # Transform mask to binary for a more accurate skeleton
    mask = load_in_mask(path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = ((mask > 0) * 1).astype('uint8')

    # Filter out small components
    #labeled = skimage.measure.label(mask)
    #props = skimage.measure.regionprops(labeled)
    #cleaned = np.zeros_like(mask)
    #for prop in props:
    #    if prop.area >= min_size:
    #        cleaned[labeled == prop.label] = 1
    #mask = cleaned.astype(np.uint8)

    # Skeletonize and reconnect
    for _ in range(3):
        mask = skimage.morphology.dilation(mask, skimage.morphology.square(3))

    skeleton = skimage.morphology.skeletonize(mask)
    skeleton_reconnected = reconnect_skeleton(skeleton, max_dist=20)
    mask[skeleton_reconnected > 0] = 1

    plants_locations = detect_roots(mask, expected_centers)

    return [plants_locations, mask]

def find_best_start_node(mask_subset_branch: pd.DataFrame, G_biased: nx.DiGraph, beta: float = 5.0) -> int:
    """
    Returns the best node to use as a starting point based on vertical position and connectivity.

    :param mask_subset_branch: DataFrame from summarize(...), filtered to bbox and label.
    :param G_biased: Biased DiGraph with weights.
    :param beta: Weighting factor for number of reachable nodes (higher = more emphasis on reach).
    :return: node-id to use as start
    """
    candidates = mask_subset_branch['node-id-src'].unique()
    id_to_y = mask_subset_branch.set_index('node-id-src')['image-coord-src-0'].to_dict()

    best_score = -np.inf
    best_node = None

    for node in candidates:
        y = id_to_y.get(node, 99999)
        try:
            reachable = len(nx.descendants(G_biased, node))
            score = -y + beta * reachable
            if score > best_score:
                best_score = score
                best_node = node
        except Exception:
            continue

    print(f"Selected start node: {best_node}, score: {best_score}")
    return best_node

def build_biased_graph_old(mask_full_branch):
    """
    Build a graph with cost equal to Euclidean distance between nodes.
    This ensures Dijkstra finds the geometrically shortest path.
    """
    G = nx.DiGraph()

    for _, row in mask_full_branch.iterrows():
        src = row['node-id-src']
        dst = row['node-id-dst']
        y0, x0 = row['image-coord-src-0'], row['image-coord-src-1']
        y1, x1 = row['image-coord-dst-0'], row['image-coord-dst-1']

        distance = math.hypot(x1 - x0, y1 - y0)

        G.add_edge(src, dst, weight=distance)
        G.add_edge(dst, src, weight=distance)  # Add reverse edge if bidirectional

    return G

def build_biased_graph(
    mask_full_branch,
    full_skeleton,
    head_length=50,
    angle_penalty=50.0,
    crop_padding=500,
    reverse_penalty_factor=5.0,
    allow_reverse=False,
    forbid_upward=True
):
    """
    Build a graph where edge cost is based on angular deviation from vertical,
    measured over the first `head_length` pixels in the path.

    Parameters:
        mask_full_branch: DataFrame with edge node and coordinate info.
        full_skeleton: 2D numpy array of the skeletonized mask.
        head_length: Number of pixels to consider when computing angle.
        angle_penalty: Weight factor for angular deviation.
        crop_padding: Padding around edge coordinates for cropping.
        reverse_penalty_factor: Cost multiplier for reverse edges.
        allow_reverse: Whether to allow reverse edges (dst → src).
        forbid_upward: Whether to prevent forward edges that go upward (dy < 0).
    """
    import math
    import numpy
    import networkx
    import skimage.graph

    G = networkx.DiGraph()

    for _, row in mask_full_branch.iterrows():
        src = row['node-id-src']
        dst = row['node-id-dst']
        y_src, x_src = int(row['image-coord-src-0']), int(row['image-coord-src-1'])
        y_dst, x_dst = int(row['image-coord-dst-0']), int(row['image-coord-dst-1'])

        G.add_node(src)
        G.add_node(dst)

        # Crop region for localized pathfinding
        y_min = max(0, min(y_src, y_dst) - crop_padding)
        y_max = min(full_skeleton.shape[0], max(y_src, y_dst) + crop_padding)
        x_min = max(0, min(x_src, x_dst) - crop_padding)
        x_max = min(full_skeleton.shape[1], max(x_src, x_dst) + crop_padding)

        sub_skeleton = full_skeleton[y_min:y_max, x_min:x_max]
        offset_src = (y_src - y_min, x_src - x_min)
        offset_dst = (y_dst - y_min, x_dst - x_min)

        try:
            path_pixels, _ = skimage.graph.route_through_array(
                1 - sub_skeleton,
                offset_src,
                offset_dst,
                fully_connected=True
            )

            if len(path_pixels) < 2:
                continue

            # Extract head segment for angle calculation
            head = path_pixels[:min(head_length, len(path_pixels))]
            if len(head) < 2:
                continue

            y_head = numpy.array([p[0] for p in head])
            x_head = numpy.array([p[1] for p in head])
            dy = y_head[-1] - y_head[0]
            dx = x_head[-1] - x_head[0]
            length = math.hypot(dx, dy)
            if length == 0:
                continue

            # Optional: skip upward movement
            if forbid_upward and dy < 0:
                continue

            cos_theta = numpy.clip(dy / length, -1.0, 1.0)
            angle_rad = math.acos(cos_theta)
            angle_deg = math.degrees(angle_rad)

            cost = angle_penalty * (angle_deg / 180.0)

            G.add_edge(src, dst,
                       cost=cost,
                       dy=dy,
                       dx=dx,
                       angle_deg=angle_deg,
                       length=length)

            if allow_reverse:
                reverse_cost = cost * reverse_penalty_factor
                G.add_edge(dst, src,
                           cost=reverse_cost,
                           dy=-dy,
                           dx=-dx,
                           angle_deg=angle_deg,
                           length=length)

        except Exception:
            continue

    return G

import math

def compute_cumulative_angle_deviation(coords):
    """
    Compute the sum of angle deviations along the path segments.
    Smaller values mean straighter path.
    """
    total_angle = 0
    for i in range(1, len(coords) - 1):
        y0, x0 = coords[i - 1]
        y1, x1 = coords[i]
        y2, x2 = coords[i + 1]

        v1 = np.array([x1 - x0, y1 - y0])
        v2 = np.array([x2 - x1, y2 - y1])

        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            continue

        # angle between vectors (in radians)
        angle = math.acos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))
        total_angle += angle

    return total_angle

def select_best_path(G, start_node, coord_map, min_dy_ratio=0.8, min_length=2, top_n=1):
    if start_node not in G:
        return []

    try:
        _, paths = nx.single_source_dijkstra(G, start_node, weight='cost')
    except nx.NetworkXNoPath:
        return []

    path_infos = []

    for node, path in paths.items():
        if node == start_node or len(path) < min_length:
            continue

        coords = [coord_map[n] for n in path if n in coord_map]
        if len(coords) < 2:
            continue

        y0, x0 = coords[0]
        y1, x1 = coords[-1]
        dy = y1 - y0
        dx = abs(x1 - x0)
        if dy <= 0:
            continue

        cumulative_angle = compute_cumulative_angle_deviation(coords)

        path_infos.append({
            'path': path,
            'dy': dy,
            'angle_deviation': cumulative_angle,
        })

    if not path_infos:
        return []

    # Step 1: Top N longest downward paths
    max_dy = max(p['dy'] for p in path_infos)
    dy_cutoff = max_dy * min_dy_ratio
    dy_candidates = [p for p in path_infos if p['dy'] >= dy_cutoff]
    dy_candidates.sort(key=lambda p: p['dy'], reverse=True)
    dy_candidates = dy_candidates[:top_n]

    # Step 2: Among them, select path with lowest total angle deviation
    best = min(dy_candidates, key=lambda p: p['angle_deviation'])

    return best['path']

def crop_skeleton_by_bbox(skeleton, bbox, scale=1.5):
    """
    Crop the skeleton image using a scaled bounding box.

    Parameters:
    - skeleton: np.ndarray, binary image of the full skeleton
    - bbox: tuple (ymin, ymax, xmin, xmax), original bounding box
    - scale: float, scaling factor to enlarge the bounding box

    Returns:
    - cropped_skeleton: np.ndarray, cropped binary skeleton image
    """
    img_height, img_width = skeleton.shape[:2]
    ymin, ymax, xmin, xmax = bbox

    # Compute center and new size
    y_center = (ymin + ymax) / 2
    x_center = (xmin + xmax) / 2
    height = (ymax - ymin) * scale
    width = (xmax - xmin) * scale

    # Calculate new coordinates
    new_ymin = max(0, int(round(y_center - height / 2)))
    new_ymax = min(img_height, int(round(y_center + height / 2)))
    new_xmin = max(0, int(round(x_center - width / 2)))
    new_xmax = min(img_width, int(round(x_center + width / 2)))

    # Crop the skeleton
    cropped_skeleton = np.zeros_like(skeleton, dtype=np.uint8)
    cropped_skeleton[new_ymin:new_ymax, new_xmin:new_xmax] = skeleton[new_ymin:new_ymax, new_xmin:new_xmax]

    return cropped_skeleton

def find_main_root_path(
    mask_subset_branch, mask_full_branch, G_biased,
    coord_map, select_best_path, min_coverage_ratio=0.8
):
    candidates = mask_subset_branch['node-id-src'].unique()

    id_to_y_subset = {
        **mask_subset_branch.set_index('node-id-src')['image-coord-src-0'].to_dict(),
        **mask_subset_branch.set_index('node-id-dst')['image-coord-dst-0'].to_dict(),
    }
    id_to_y_full = {
        **mask_full_branch.set_index('node-id-src')['image-coord-src-0'].to_dict(),
        **mask_full_branch.set_index('node-id-dst')['image-coord-dst-0'].to_dict(),
    }

    y_values = mask_subset_branch['image-coord-src-0'].tolist() + mask_subset_branch['image-coord-dst-0'].tolist()
    total_root_height = np.ptp(y_values)

    best_path = []
    best_start_node = None
    best_coverage = -1

    candidates_sorted = sorted(candidates, key=lambda n: id_to_y_subset.get(n, float('inf')))

    for node in candidates_sorted:
        try:
            path = select_best_path(G_biased, node, coord_map)
            if not path or len(path) < 2:
                print(f"⚠️ Path from node {node} is empty or too short.")
                continue

            y_start = id_to_y_subset.get(path[0], 0)
            y_end = id_to_y_full.get(path[-1], 0)
            path_height = y_end - y_start
            coverage_ratio = path_height / total_root_height if total_root_height > 0 else 0

            print(f"🔎 Node {node}: path_height = {path_height}")

            if coverage_ratio > best_coverage:
                best_path = path
                best_start_node = node
                best_coverage = coverage_ratio
                print(f"✔️ Updated best path from node {node}")

        except Exception as e:
            print(f"❌ Error with node {node}: {e}")

    if not best_path:
        print("❌ No valid path found from any candidate.")
        return None, []

    if best_coverage < min_coverage_ratio:
        print(f"⚠️ No path met the coverage threshold ({min_coverage_ratio:.0%}). Returning best fallback with {best_coverage:.1%}.")

    print(f"✅ Selected best path from node {best_start_node} with coverage {best_coverage:.1%}")
    return best_start_node, best_path

def segmentation_primary(mask_path, expected_centers=expected_centers, reconnect_graph_max_dist: float = 20.0):

    def normalize_pair(src, dst):
        return tuple(sorted((src, dst)))

    def crop_within_bounds(img, bbox, scale=1.5):
        h, w = img.shape[:2]
        ymin, ymax, xmin, xmax = bbox
        y_center = (ymin + ymax) / 2
        x_center = (xmin + xmax) / 2
        half_h = (ymax - ymin) * scale / 2
        half_w = (xmax - xmin) * scale / 2
        new_ymin = max(0, int(y_center - half_h))
        new_ymax = min(h, int(y_center + half_h))
        new_xmin = max(0, int(x_center - half_w))
        new_xmax = min(w, int(x_center + half_w))
        cropped = np.zeros_like(img)
        cropped[new_ymin:new_ymax, new_xmin:new_xmax] = img[new_ymin:new_ymax, new_xmin:new_xmax]
        return cropped

    print(f"Mask path: {mask_path}")
    plant_location, mask = prepare_data_for_segmentation(mask_path, expected_centers)
    full_skeleton = skimage.morphology.skeletonize(mask)

    if len(np.unique(full_skeleton)) == 1:
        warnings.warn("Given mask is empty")
        return None

    mask_full_skeleton_ob = Skeleton(full_skeleton)
    mask_full_branch = summarize(mask_full_skeleton_ob)
    mask_full_branch['root_type'] = "None"
    mask_full_branch['plant'] = "None"

    G_biased = build_biased_graph(mask_full_branch, full_skeleton)

    coord_map = {row['node-id-src']: (row['image-coord-src-0'], row['image-coord-src-1']) for _, row in mask_full_branch.iterrows()}
    coord_map.update({row['node-id-dst']: (row['image-coord-dst-0'], row['image-coord-dst-1']) for _, row in mask_full_branch.iterrows()})

    # Assign labels
    components = list(nx.connected_components(G_biased.to_undirected()))
    label_map = {}
    for i, comp in enumerate(components):
        for node in comp:
            label_map[node] = i

    tips_by_label = {}
    for node in G_biased.nodes:
        if G_biased.degree(node) <= 2 and node in coord_map:
            label = label_map[node]
            tips_by_label.setdefault(label, []).append(node)

    virtual_edges = []

    for label1, label2 in combinations(tips_by_label.keys(), 2):
        tips1 = tips_by_label[label1]
        tips2 = tips_by_label[label2]
        if not tips1 or not tips2:
            continue
        best_dist = np.inf
        best_pair = (None, None)
        for tip1 in tips1:
            for tip2 in tips2:
                coord1 = np.array(coord_map[tip1])
                coord2 = np.array(coord_map[tip2])
                dist = np.linalg.norm(coord2 - coord1)
                if dist < best_dist:
                    best_dist = dist
                    best_pair = (tip1, tip2)
        if best_dist > reconnect_graph_max_dist:
            continue
        tip1, tip2 = best_pair
        y1, _ = coord_map[tip1]
        y2, _ = coord_map[tip2]
        if not (G_biased.has_edge(tip1, tip2) or G_biased.has_edge(tip2, tip1)):
            if y1 < y2:
                G_biased.add_edge(tip1, tip2, score=-best_dist, virtual=True)
                direction = f"{tip1} → {tip2}"
            else:
                G_biased.add_edge(tip2, tip1, score=-best_dist, virtual=True)
                direction = f"{tip2} → {tip1}"
            virtual_edges.append((coord_map[tip1], coord_map[tip2]))
            print(f"🔧 Added downward virtual edge {direction}, dist={best_dist:.2f}")

    num_virtual_edges = sum(1 for u, v, d in G_biased.edges(data=True) if d.get("virtual", False))
    print(f"✅ Total virtual edges added: {num_virtual_edges}")

    virtual_edges_image = np.stack([mask]*3, axis=-1).astype(np.uint8) * 255
    for (y1, x1), (y2, x2) in virtual_edges:
        rr, cc = line(int(y1), int(x1), int(y2), int(x2))
        for r, c in zip(rr, cc):
            rr_d, cc_d = skimage.draw.disk((r, c), radius=5, shape=virtual_edges_image.shape[:2])
            virtual_edges_image[rr_d, cc_d] = [255, 0, 0]

    for plant_num, bbox in enumerate(plant_location):
        print(f"Processing plant #{plant_num}")
        cropped_skel = crop_within_bounds(full_skeleton, bbox, scale=1)
        coords = set(zip(*np.where(cropped_skel > 0)))

        # def row_in_crop(row):
        #     return (row['image-coord-src-0'], row['image-coord-src-1']) in coords and \
        #            (row['image-coord-dst-0'], row['image-coord-dst-1']) in coords

        def row_in_crop(row):
            return (row['image-coord-src-0'], row['image-coord-src-1']) in coords

        mask_subset_branch = mask_full_branch[mask_full_branch.apply(row_in_crop, axis=1)].copy()
        if mask_subset_branch.empty:
            continue

        # build subset of the graph
        # Identify node IDs from crop region
        subset_nodes = set(mask_subset_branch['node-id-src']).union(mask_subset_branch['node-id-dst'])

        # Manually build the subgraph to include any edge that connects:
        # - two subset nodes, or
        # - one subset node and one node connected via a virtual edge
        edges_to_include = []
        for u, v, data in G_biased.edges(data=True):
            if u in subset_nodes or v in subset_nodes:
                if data.get("virtual", False):
                    # virtual edge: include if one node is in the region
                    edges_to_include.append((u, v, data))
                elif u in subset_nodes and v in subset_nodes:
                    # regular edge: include only if both nodes are in the region
                    edges_to_include.append((u, v, data))

        # Build the subset graph manually
        G_biased_subset = nx.DiGraph()
        G_biased_subset.add_edges_from(edges_to_include)

        # find main root
        start_node, node_id_path = find_main_root_path(
            mask_subset_branch,
            mask_full_branch,
            G_biased,
            coord_map,  # ✅ pass it explicitly
            select_best_path
        )

        main_root_pairs = set(zip(node_id_path[:-1], node_id_path[1:]))

        for idx, row in mask_full_branch.iterrows():
            pair = (row["node-id-src"], row["node-id-dst"])
            if pair in main_root_pairs:
                mask_full_branch.loc[idx, 'root_type'] = "Primary"
                mask_full_branch.loc[idx, 'plant'] = plant_num

        pair_to_indices = {}
        for idx, row in mask_full_branch[mask_full_branch['root_type'] == 'Primary'].iterrows():
            norm_pair = normalize_pair(row["node-id-src"], row["node-id-dst"])
            pair_to_indices.setdefault(norm_pair, []).append(idx)

        for indices in pair_to_indices.values():
            if len(indices) > 1:
                # 🔽 Keep the one with the shortest branch-distance
                best_idx = min(indices, key=lambda i: mask_full_branch.loc[i, 'branch-distance'])
                for idx in indices:
                    if idx != best_idx:
                        mask_full_branch.loc[idx, 'root_type'] = "None"

    return [mask_full_branch, mask_full_skeleton_ob, plant_location, virtual_edges_image]

def getting_coords_for_starting_nodes(branch: pd.core.frame.DataFrame) -> List[Tuple[int, int]]:
    """
    Finds coordinates for all the starting nodes in the branch

    Author: Fedya Chursin, 220904@buas.nl

    :param branch: DataFrame with all the branches in it (graph representation).
    :type branch: pd.DataFrame
    :return: List of tuples with starting nodes coordinates (y, x).
    :rtype: List[Tuple[int, int]]
    """
    starting_nodes_list = determine_starting_nodes(branch)
    start_nodes_coordinates = []
    for id in starting_nodes_list:
        x = branch[branch['node-id-src'] == id]['image-coord-src-1'].iloc[0]
        y = branch[branch['node-id-src'] == id]['image-coord-src-0'].iloc[0]
        start_nodes_coordinates.append((y, x))
    return start_nodes_coordinates

def determine_starting_nodes(branch: pd.DataFrame) -> List[int]:
    """
    Determines all the starting points in the branch

    Author: Fedya Chursin, 220904@buas.nl

    :param branch: DataFrame summarizing the skeleton of the branch with columns 'node-id-src' and 'node-id-dst'.
    :type branch: pd.DataFrame
    :return: List of node IDs that are starting points.
    :rtype: List[int]
    """
    # get uniques src and dst nodes
    src_nodes = branch['node-id-src'].unique()
    dst_nodes = branch['node-id-dst'].unique()
    starting_nodes = []
    # loop through all src nodes if it's not in dst nodes -> append to starting nodes array
    for node in src_nodes:
        if node not in dst_nodes:
            starting_nodes.append(node)
    return starting_nodes

def get_lateral(branch_loc: pd.DataFrame, subset_skeleton_ob_arr: Skeleton, img: np.ndarray, plant_bboxes: List[Tuple[int, int, int, int]]) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Assigns lateral roots to plants using a two-step method:
    1. Follow lateral paths from primary roots (no bbox constraint).
    2. Assign remaining unassigned branches to plants based on their position in bounding boxes.

    Modifications:
    - Works in-place on subset_branch (no copies).
    - Skips already assigned primary roots or assigned segments.
    - Assigns root_id as 'Lateral_X' based on counter.
    """

    img2 = img.copy()
    subset_branch = branch_loc.copy()

    # Step 1: Follow lateral paths connected to primary roots (no bbox)
    subset_branch["angle"] = subset_branch.apply(calculate_angle, axis=1)
    subset_branch["root_id"] = np.nan
    subset_branch["root_id"] = subset_branch["root_id"].astype(object)
    subset_branch = follow_lateral_path(subset_branch)

    # Step 2: Assign remaining unassigned segments based on bbox
    # Extract numeric part of existing root_ids
    existing_ids = subset_branch["root_id"].dropna().astype(str)
    existing_numbers = [int(re.findall(r"\d+", rid)[0]) for rid in existing_ids if re.findall(r"\d+", rid)]
    root_id_counter = max(existing_numbers) + 1 if existing_numbers else 0

    for idx, row in subset_branch.iterrows():
        if row.get("root_type") == "Primary":
            continue
        if pd.notna(row.get("root_id")):
            continue

        coords = subset_skeleton_ob_arr.path_coordinates(idx)
        if len(coords) == 0:
            continue

        y0, x0 = coords[0]

        for plant_idx, (ymin, ymax, xmin, xmax) in enumerate(plant_bboxes):
            if ymin <= y0 <= ymax and xmin <= x0 <= xmax:
                subset_branch.at[idx, "plant"] = plant_idx
                subset_branch.at[idx, "root_type"] = "Lateral"
                subset_branch.at[idx, "root_id"] = f"Lateral_{root_id_counter}"
                root_id_counter += 1
                break  # Stop after assigning to one plant

    # Visualization
    colors = [(0, 128, 255), (240, 32, 160), (255, 0, 0), (255, 0, 255), (0, 255, 0)]
    img2 = draw_lateral(subset_branch, subset_skeleton_ob_arr, img2, colors)

    return img2, subset_branch

def find_shoot(
    shoot_mask: np.ndarray,
    expected_centers,
    initial_box_halfsize_x: int = 200,
    initial_box_halfsize_y: int = 300,
    expansion_step: int = 1,
    max_empty_expansions: int = 200,
    max_total_expansion: int = 5000
) -> list[np.ndarray]:
    """
    Expands from expected centers and returns a list of shoot masks,
    one per plant region, without merging them.
    """
    shoot_mask = ((shoot_mask > 0) * 1).astype(np.uint8)
    shoot_mask = shoot_mask[:, :, 1] if shoot_mask.ndim == 3 else shoot_mask
    height, width = shoot_mask.shape
    shoot_masks = []
    final_boxes = []

    def filter_connected_components(mask, min_size=50):
        labels = skimage.measure.label(mask)
        props = skimage.measure.regionprops(labels)
        cleaned = np.zeros_like(mask)
        for prop in props:
            if prop.area >= min_size:
                cleaned[labels == prop.label] = 1
        return cleaned.astype(np.uint8)

    shoot_mask = filter_connected_components(shoot_mask, min_size=50)

    def overlaps_any(new_box, existing_boxes):
        y0, y1, x0, x1 = new_box
        for ey0, ey1, ex0, ex1 in existing_boxes:
            if not (x1 <= ex0 or x0 >= ex1 or y1 <= ey0 or y0 >= ey1):
                return True
        return False

    for cx, cy, _ in expected_centers:
        ymin = max(cy - initial_box_halfsize_y, 0)
        ymax = min(cy + initial_box_halfsize_y, height)
        xmin = max(cx - initial_box_halfsize_x, 0)
        xmax = min(cx + initial_box_halfsize_x, width)

        empty_counts = {"up": 0, "down": 0, "left": 0, "right": 0}
        can_expand = {"up": True, "down": True, "left": True, "right": True}
        total_expansion = 0

        while any(can_expand.values()) and total_expansion < max_total_expansion:
            grew = False

            for direction in ["up", "down", "left", "right"]:
                if not can_expand[direction]:
                    continue

                if direction == "up":
                    y0 = max(ymin - expansion_step, 0)
                    new_strip = shoot_mask[y0:ymin, xmin:xmax]
                    box = [y0, ymax, xmin, xmax]
                elif direction == "down":
                    y1 = min(ymax + expansion_step, height)
                    new_strip = shoot_mask[ymax:y1, xmin:xmax]
                    box = [ymin, y1, xmin, xmax]
                elif direction == "left":
                    x0 = max(xmin - expansion_step, 0)
                    new_strip = shoot_mask[ymin:ymax, x0:xmin]
                    box = [ymin, ymax, x0, xmax]
                elif direction == "right":
                    x1 = min(xmax + expansion_step, width)
                    new_strip = shoot_mask[ymin:ymax, xmax:x1]
                    box = [ymin, ymax, xmin, x1]
                else:
                    continue

                if np.any(new_strip) and not overlaps_any(box, final_boxes):
                    if direction == "up":
                        ymin = y0
                    elif direction == "down":
                        ymax = y1
                    elif direction == "left":
                        xmin = x0
                    elif direction == "right":
                        xmax = x1

                    empty_counts[direction] = 0
                    grew = True
                    total_expansion += expansion_step
                else:
                    empty_counts[direction] += 1
                    if empty_counts[direction] >= max_empty_expansions:
                        can_expand[direction] = False

        final_boxes.append([ymin, ymax, xmin, xmax])

        # Isolate region and store mask
        region = shoot_mask[ymin:ymax, xmin:xmax]
        shoot_i = np.zeros_like(shoot_mask)
        shoot_i[ymin:ymax, xmin:xmax] = region
        shoot_masks.append(shoot_i)

    return shoot_masks

def increment_if_numeric(val):
    try:
        return int(val) + 1
    except (ValueError, TypeError):
        return val

def get_root_tips(binary_mask):
    skeleton = skeletonize(binary_mask > 0)
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]])
    neighbor_count = convolve(skeleton.astype(np.uint8), kernel, mode='constant',
                              cval=0)
    tips = np.logical_and(skeleton, neighbor_count == 11)  # 10(center) + 1 neighbor
    return tips.astype(np.uint8) * 255

def measure_folder(folder_dir, expected_centers) -> pd.DataFrame:
        """
        This function checks for each timeline, each petri dish. It uses the binary root masks for this.
        It starts with segmenting the roots, and finding the primary roots. After this, it uses the get_lateral function
        to find the lateral roots of every plant. It saves the primary, lateral, and total root length of each plant into an Excel file.

        :param folder_dir: Directory with the location of all the timeline folders.
        """
        # Loop through all the timelines
        for timeline in os.listdir(folder_dir):
            print(f"TIMELINE:{timeline}")
            if ".DS_Store" not in timeline:
                # Loop through all the petri dishes in each timeline
                for petri_dish in track(os.listdir(f"{folder_dir}/{timeline}"),
                                        description=f' Segmenting and Measuring plants for petri dish - {timeline}'):
                    if petri_dish.endswith((".png", ".jxl")):
                        path_to_masks = f"{folder_dir}/{timeline}/{petri_dish[:-4]}"
                        print(path_to_masks)
                        # # Skip if already processed
                        # if os.path.exists(f"{path_to_masks}/measurements.xlsx"):
                        #     continue

                        if ("flagged.txt" in os.listdir(f"{folder_dir}/{timeline}/{petri_dish[:-4]}")):
                            shoot_mask = f"{path_to_masks}/shoot_mask.png"
                            shoot_mask = cv2.imread(shoot_mask, 0)
                            root_mask = f"{path_to_masks}/root_mask_fixed.png"
                            root_mask = cv2.imread(root_mask, 0)
                            image = cv2.imread(f"{folder_dir}/{timeline}/{petri_dish}")
                            # image = cv2.flip(image, 1)
                            new_image = add_shoot_image_mask(image, shoot_mask)
                            new_image = add_shoot_image_mask(new_image, root_mask)
                            cv2.imwrite(f"{path_to_masks}/image_mask.png", new_image)

                    try:
                        # Check if the file is a png or jxl file
                        if petri_dish.endswith((".png", ".jxl")) and not os.path.isfile(f"{folder_dir}/{timeline}/{timeline}/flagged.txt"):
                            # Get the path to the petri dish
                            path_to_petri_dish = f"{folder_dir}/{timeline}/{petri_dish}"
                            # Get the path to the root masks
                            path_to_masks = f"{folder_dir}/{timeline}/{petri_dish[:-4]}"
                            # Get the path to the root mask
                            for mask in os.listdir(path_to_masks):
                                # Make sure that the mask is the root mask
                                if mask.endswith("root_mask_fixed.png"):
                                    mask_path = f"{path_to_masks}/{mask}"
                                elif mask.endswith("shoot_mask.png"):
                                    shoot_mask_path = f"{path_to_masks}/{mask}"
                            if petri_dish.endswith(".jxl"):
                                # Read the petri dish image.
                                image = Image.open(path_to_petri_dish)
                                image = np.array(image)

                                if len(image.shape) == 3 and image.shape[2] == 4:
                                    image = image[:, :, :3]

                            else:
                                # Read the root image
                                image = cv2.imread(path_to_petri_dish)
                                # Convert the image to RGB
                                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                            # Perform segmentation and find the primary roots
                            primary_root_segmentation = segmentation_primary(mask_path, expected_centers=expected_centers)
                            if primary_root_segmentation is not None:
                                mask_full_branch, skelton_ob_loc, plant_location, virtual_edges_image = primary_root_segmentation
                                # Draw the primary roots on the image
                                for index, row in mask_full_branch.iterrows():
                                    if row["root_type"] == "Primary":
                                        yx = skelton_ob_loc.path_coordinates(index)
                                        image = draw_root(yx, image, 0, 0, (0, 0, 255))
                                # Find the lateral roots
                                mask_full_branch.to_csv("branch_loc.csv", index=False)
                                # get_lateral recolor the image according to the lateral roots
                                image, mask_full_branch = get_lateral(
                                    mask_full_branch, skelton_ob_loc, image, plant_location)
                                mask_full_branch["plant"] = mask_full_branch["plant"].apply(increment_if_numeric)

                                # Set the root_id to primary for each primary root type
                                mask_full_branch.loc[mask_full_branch['root_type'] == 'Primary', 'root_id'] = 'Primary'

                                # load shoot mask and apply bboxes method identifcation
                                shoot_mask = cv2.imread(shoot_mask_path)
                                new_shoot_mask = find_shoot(shoot_mask, expected_centers = expected_centers)
                                for ind_shoot in new_shoot_mask:
                                    ind_shoot = ind_shoot.astype(bool)
                                    image[ind_shoot] = (0, 100, 0)

                                # Save the segmented image
                                cv2.imwrite(f'{path_to_masks}/image_mask.png', image)

                                # save rsml format
                                rsml.get_rsml(mask_full_branch, skelton_ob_loc, f"{path_to_masks}/root_structure.rsml")

                                # Create a DataFrame to store the measurements
                                measurement_df = pd.DataFrame(
                                    columns=["plant",
                                             "Primary_length(px)", "Primary_length(mm)",
                                             "Lateral_length(px)", "Lateral_length(mm)",
                                             "Total_length(px)", "Total_length(mm)",
                                             "Leaf_size(px)", "Lateral_root_count"]
                                )

                                global_landmark_df = pd.DataFrame()
                                # Loop over all the plants in the subset_branch to get the primary, lateral, and total root length of each plant
                                mask_full_branch = mask_full_branch[mask_full_branch["plant"] != "None"]
                                for x in mask_full_branch["plant"].unique():
                                    if math.isnan(x):
                                        continue
                                    img_mask = np.zeros_like(image)
                                    landmarked_img = image.copy()
                                    plant_df = pd.DataFrame(
                                        columns=["plant", "Primary_length(px)", "Lateral_length(px)", "Total_length(px)", "Leaf_size(px)", "Lateral_root_count"])

                                    os.makedirs(f"{path_to_masks}/plant_{x}", exist_ok=True)

                                    # Skip the plants that are not assigned to any plant
                                    subset_plant = mask_full_branch[mask_full_branch["plant"] == x]
                                    mask_full_branch.loc[subset_plant.index, "connected_to_main_root"] = False

                                    subset_plant.to_excel(f'{path_to_masks}/plant_{x}/plant_data.xlsx')
                                    # Filter the DataFrame to keep the primary and lateral root lengths
                                    subset_plant_primary = subset_plant[subset_plant["root_type"] == "Primary"]
                                    subset_plant_lateral = subset_plant[subset_plant["root_type"] == "Lateral"]
                                    # Calculate the primary, lateral, and total root length of the plant
                                    primary_length = subset_plant_primary["branch-distance"].sum()
                                    lateral_length = subset_plant_lateral["branch-distance"].sum()
                                    # Calculate the total root length of the plant
                                    total_length = primary_length + lateral_length

                                    main_root_nodes = set()
                                    for idx in subset_plant_primary.index:
                                        main_root_nodes.update(map(tuple, skelton_ob_loc.path_coordinates(idx)))
                                    lateral_count = 0
                                    for idx in subset_plant_lateral.index:
                                        path_nodes = set(map(tuple, skelton_ob_loc.path_coordinates(idx)))
                                        if path_nodes & main_root_nodes:
                                            lateral_count += 1
                                            mask_full_branch.loc[idx, "connected_to_main_root"] = True

                                    for index, row in subset_plant.iterrows():
                                        yx = skelton_ob_loc.path_coordinates(index)
                                        img_mask = draw_root(yx, img_mask, 0, 0, (255, 255, 255))

                                    cv2.imwrite(f'{path_to_masks}/plant_{x}/root_mask.png', img_mask)

                                    # Generate primary/lateral masks
                                    primary_mask = np.zeros_like(img_mask[:, :, 0])
                                    lateral_mask = np.zeros_like(img_mask[:, :, 0])

                                    for index, row in subset_plant.iterrows():
                                        yx = skelton_ob_loc.path_coordinates(index)
                                        if row["root_type"] == "Primary":
                                            primary_mask = draw_root(yx, primary_mask, 0, 0, 255)
                                        elif row["root_type"] == "Lateral":
                                            lateral_mask = draw_root(yx, lateral_mask, 0, 0, 255)

                                    cv2.imwrite(f'{path_to_masks}/plant_{x}/main_root_mask.png', primary_mask)
                                    cv2.imwrite(f'{path_to_masks}/plant_{x}/lateral_root_mask.png', lateral_mask)

                                    # generate tips and node mask
                                    fused_mask = primary_mask.copy()
                                    fused_mask[lateral_mask == 255] = 255
                                    fused_tips = get_root_tips(fused_mask)
                                    lateral_tips = get_root_tips(lateral_mask)
                                    node_mask = np.where((fused_tips == 0) & (lateral_tips == 255), 255, 0).astype(np.uint8)

                                    cv2.imwrite(f'{path_to_masks}/plant_{x}/tip_mask.png', fused_tips)
                                    cv2.imwrite(f'{path_to_masks}/plant_{x}/node_mask.png', node_mask)

                                    # Shoot-related masks
                                    if 0 <= (x - 1) < len(new_shoot_mask):
                                        shoot_mask_x = new_shoot_mask[x - 1]
                                        shoot_mask_x = shoot_mask_x * 255
                                        shoot_root_mask = merge_shoot_root(shoot_mask_x, img_mask)
                                        image = add_shoot_image_mask(image, shoot_mask_x)
                                        cv2.imwrite(f'{path_to_masks}/plant_{x}/shoot_mask.png', shoot_mask_x)
                                        cv2.imwrite(f'{path_to_masks}/plant_{x}/shoot_root_mask.png', shoot_root_mask)
                                    else:
                                        shoot_mask_x = np.zeros_like(img_mask[:, :, 0])  # fallback empty mask

                                    # Compute leaf size
                                    leaf_size = int(np.sum(shoot_mask_x > 0))  # number of white pixels

                                    primary_length = subset_plant_primary["branch-distance"].sum()
                                    lateral_length = subset_plant_lateral["branch-distance"].sum()
                                    total_length = primary_length + lateral_length

                                    # Convert to millimeters
                                    primary_length_mm = apply_conversion_factor(primary_length)
                                    lateral_length_mm = apply_conversion_factor(lateral_length)
                                    total_length_mm = apply_conversion_factor(total_length)

                                    new_row = {
                                        "plant": x,
                                        "Primary_length(px)": primary_length,
                                        "Primary_length(mm)": primary_length_mm,
                                        "Lateral_length(px)": lateral_length,
                                        "Lateral_length(mm)": lateral_length_mm,
                                        "Total_length(px)": total_length,
                                        "Total_length(mm)": total_length_mm,
                                        "Leaf_size(px)": leaf_size,
                                        "Lateral_root_count": lateral_count
                                    }

                                    plant_df.loc[len(plant_df)] = new_row
                                    plant_df.to_excel(f'{path_to_masks}/plant_{x}/plant_measurements.xlsx')
                                    # Add the new row to the DataFrame
                                    measurement_df.loc[len(measurement_df)] = new_row

                                    # Landmark export
                                    subset_plant_clean = subset_plant.drop(columns=["connected_to_main_root"], errors="ignore").copy()
                                    landmark_df = get_landmarks(subset_plant_clean)
                                    landmarked_img = draw_landmarks(landmarked_img, landmark_df)
                                    primary_measurements = subset_plant[subset_plant["root_type"] == "Primary"]
                                    primary_measurements.to_excel(f'{path_to_masks}/plant_{x}/primary_measurements.xlsx')

                                    cv2.imwrite(f'{path_to_masks}/plant_{x}/landmarked_image.png', landmarked_img)
                                    landmark_df[["landmark", "y", "x", "root_id"]].to_excel(
                                        f'{path_to_masks}/plant_{x}/landmarks.xlsx')

                                    landmark_df["plant"] = x
                                    global_landmark_df = pd.concat([global_landmark_df, landmark_df[["landmark", "plant", "root_id", "y", "x"]]], ignore_index=True)

                                # Save the measurements to an Excel file
                                measurement_df.to_excel(f'{path_to_masks}/measurements.xlsx')
                                global_landmark_df = global_landmark_df[global_landmark_df["root_id"] == "Primary"]
                                global_landmark_df[["landmark", "plant", "x", "y"]].sort_values(by="plant").to_excel(f'{path_to_masks}/landmarks.xlsx')
                                cv2.imwrite(f'{path_to_masks}/image_mask.png', image)
                            else:
                                measurement_df = pd.DataFrame(
                                    columns=[
                                        "plant",
                                        "Primary_length(px)", "Primary_length(mm)",
                                        "Lateral_length(px)", "Lateral_length(mm)",
                                        "Total_length(px)", "Total_length(mm)",
                                        "Leaf_size(px)", 'Lateral_root_count'])
                                new_row = {"plant": 0, "Primary_length(mm)": 0,
                                           "Lateral_length(mm)": 0, "Total_length(mm)": 0, "Leaf_size(px)": 0, 'Lateral_root_count':0}
                                measurement_df.loc[len(measurement_df)] = new_row
                                measurement_df.to_excel(f'{path_to_masks}/measurements.xlsx')
                    except Exception as e:
                        print(f"Error processing plant: {e}")
                        measurement_df = pd.DataFrame(
                            columns=["plant", "Primary_length(mm)", "Lateral_length(mm)", "Total_length(mm)", "Leaf_size(px)", 'Lateral_root_count'])
                        new_row = {"plant": 0, "Primary_length(mm)": 0,
                                   "Lateral_length(mm)": 0, "Total_length(mm)": 0, "Leaf_size(px)": 0, 'Lateral_root_count':0}
                        measurement_df.loc[len(measurement_df)] = new_row
                        measurement_df.to_excel(f'{path_to_masks}/measurements.xlsx')


#### MOVE TO ANOTHER FILE LATER
def save_measurements(mask_full_branch, skelton_ob_loc, plant_location, image, path_to_masks, expected_centers):
    global_landmark_df = pd.DataFrame(columns=["landmark", "plant", "root_id", "y", "x"])
    # Draw primary roots on image.
    for index, row in mask_full_branch.iterrows():
        if row["root_type"] == "Primary":
            yx = skelton_ob_loc.path_coordinates(index)
            image = draw_root(yx, image, 0, 0, (0, 0, 255))

    mask_full_branch.to_csv("branch_loc.csv", index=False)
    image, mask_full_branch = get_lateral(mask_full_branch, skelton_ob_loc, image, plant_location)
    # mask_full_branch["plant"] = mask_full_branch["plant"].apply(increment_if_numeric)
    mask_full_branch.loc[mask_full_branch['root_type'] == 'Primary', 'root_id'] = 'Primary'

    shoot_mask = cv2.imread(f"{path_to_masks}/shoot_mask.png")
    # new_shoot_mask = find_shoot(shoot_mask, expected_centers=expected_centers)
    if shoot_mask is None:
        print(f"Warning: shoot_mask.png not found at {path_to_masks}, skipping shoot detection.")
        new_shoot_mask = [np.zeros(image.shape[:2], dtype=np.uint8)] * 5
    else:
        new_shoot_mask = find_shoot(shoot_mask, expected_centers=expected_centers)

    for ind_shoot in new_shoot_mask:
        ind_shoot = ind_shoot.astype(bool)
        image[ind_shoot] = (0, 100, 0)

    cv2.imwrite(f'{path_to_masks}/image_mask.png', image)
    rsml.get_rsml(mask_full_branch, skelton_ob_loc, f"{path_to_masks}/root_structure.rsml")

    measurement_df = pd.DataFrame(columns=[
        "plant", "Primary_length(px)", "Primary_length(mm)",
        "Lateral_length(px)", "Lateral_length(mm)",
        "Total_length(px)", "Total_length(mm)",
        "Leaf_size(px)", "Lateral_root_count"
    ])
    # global_landmark_df = pd.DataFrame()
    mask_full_branch = mask_full_branch[mask_full_branch["plant"] != "None"]

    for x in mask_full_branch["plant"].unique():
        if math.isnan(x):
            continue
        img_mask = np.zeros_like(image)
        landmarked_img = image.copy()
        plant_df = pd.DataFrame(columns=["plant", "Primary_length(px)", "Lateral_length(px)", "Total_length(px)", "Leaf_size(px)", "Lateral_root_count"])
        os.makedirs(f"{path_to_masks}/plant_{x}", exist_ok=True)

        subset_plant = mask_full_branch[mask_full_branch["plant"] == x]
        mask_full_branch.loc[subset_plant.index, "connected_to_main_root"] = False
        subset_plant.to_excel(f'{path_to_masks}/plant_{x}/plant_data.xlsx')

        subset_plant_primary = subset_plant[subset_plant["root_type"] == "Primary"]
        subset_plant_lateral = subset_plant[subset_plant["root_type"] == "Lateral"]
        primary_length = subset_plant_primary["branch-distance"].sum()
        lateral_length = subset_plant_lateral["branch-distance"].sum()
        total_length = primary_length + lateral_length

        main_root_nodes = set()
        for idx in subset_plant_primary.index:
            main_root_nodes.update(map(tuple, skelton_ob_loc.path_coordinates(idx)))
        lateral_count = 0
        for idx in subset_plant_lateral.index:
            path_nodes = set(map(tuple, skelton_ob_loc.path_coordinates(idx)))
            if path_nodes & main_root_nodes:
                lateral_count += 1
                mask_full_branch.loc[idx, "connected_to_main_root"] = True

        for index, row in subset_plant.iterrows():
            yx = skelton_ob_loc.path_coordinates(index)
            img_mask = draw_root(yx, img_mask, 0, 0, (255, 255, 255))

        cv2.imwrite(f'{path_to_masks}/plant_{x}/root_mask.png', img_mask)

        primary_mask = np.zeros_like(img_mask[:, :, 0])
        lateral_mask = np.zeros_like(img_mask[:, :, 0])
        for index, row in subset_plant.iterrows():
            yx = skelton_ob_loc.path_coordinates(index)
            if row["root_type"] == "Primary":
                primary_mask = draw_root(yx, primary_mask, 0, 0, 255)
            elif row["root_type"] == "Lateral":
                lateral_mask = draw_root(yx, lateral_mask, 0, 0, 255)

        cv2.imwrite(f'{path_to_masks}/plant_{x}/main_root_mask.png', primary_mask)
        cv2.imwrite(f'{path_to_masks}/plant_{x}/lateral_root_mask.png', lateral_mask)

        fused_mask = primary_mask.copy()
        fused_mask[lateral_mask == 255] = 255
        fused_tips = get_root_tips(fused_mask)
        lateral_tips = get_root_tips(lateral_mask)
        node_mask = np.where((fused_tips == 0) & (lateral_tips == 255), 255, 0).astype(np.uint8)
        cv2.imwrite(f'{path_to_masks}/plant_{x}/tip_mask.png', fused_tips)
        cv2.imwrite(f'{path_to_masks}/plant_{x}/node_mask.png', node_mask)

        if 0 <= (x - 1) < len(new_shoot_mask):
            shoot_mask_x = new_shoot_mask[x - 1] * 255
            shoot_root_mask = merge_shoot_root(shoot_mask_x, img_mask)
            image = add_shoot_image_mask(image, shoot_mask_x)
            cv2.imwrite(f'{path_to_masks}/plant_{x}/shoot_mask.png', shoot_mask_x)
            cv2.imwrite(f'{path_to_masks}/plant_{x}/shoot_root_mask.png', shoot_root_mask)
        else:
            shoot_mask_x = np.zeros_like(img_mask[:, :, 0])

        leaf_size = int(np.sum(shoot_mask_x > 0))
        primary_length_mm = apply_conversion_factor(primary_length)
        lateral_length_mm = apply_conversion_factor(lateral_length)
        total_length_mm = apply_conversion_factor(total_length)

        new_row = {
            "plant": x,
            "Primary_length(px)": primary_length,
            "Primary_length(mm)": primary_length_mm,
            "Lateral_length(px)": lateral_length,
            "Lateral_length(mm)": lateral_length_mm,
            "Total_length(px)": total_length,
            "Total_length(mm)": total_length_mm,
            "Leaf_size(px)": leaf_size,
            "Lateral_root_count": lateral_count
        }

        plant_df.loc[len(plant_df)] = new_row
        plant_df.to_excel(f'{path_to_masks}/plant_{x}/plant_measurements.xlsx')
        measurement_df.loc[len(measurement_df)] = new_row

        subset_plant_clean = subset_plant.drop(columns=["connected_to_main_root"], errors="ignore").copy()
        landmark_df = get_landmarks(subset_plant_clean)
        if landmark_df.empty:
            print(f"Warning: no landmarks for plant {x}, skipping.")
            subset_plant_primary.to_excel(os.path.join(path_to_masks, f"plant_{x}", "primary_measurements.xlsx"))
            cv2.imwrite(os.path.join(path_to_masks, f"plant_{x}", "landmarked_image.png"), landmarked_img)
            landmark_df.to_excel(os.path.join(path_to_masks, f"plant_{x}", "landmarks.xlsx"))
            continue
        landmarked_img = draw_landmarks(landmarked_img, landmark_df)
        subset_plant_primary.to_excel(f'{path_to_masks}/plant_{x}/primary_measurements.xlsx')
        cv2.imwrite(f'{path_to_masks}/plant_{x}/landmarked_image.png', landmarked_img)
        landmark_df[["landmark", "y", "x", "root_id"]].to_excel(f'{path_to_masks}/plant_{x}/landmarks.xlsx')
        landmark_df["plant"] = x
        global_landmark_df = pd.concat([global_landmark_df, landmark_df[["landmark", "plant", "root_id", "y", "x"]]], ignore_index=True)

    measurement_df.to_excel(f'{path_to_masks}/measurements.xlsx')
    # Perform checks for global_landmark_df.
    global_landmark_df = global_landmark_df[global_landmark_df["root_id"] == "Primary"]
    if not global_landmark_df.empty:
        global_landmark_df = global_landmark_df[global_landmark_df["root_id"] == "Primary"]
        global_landmark_df[["landmark", "plant", "x", "y"]].sort_values(by="plant").to_excel(
            f'{path_to_masks}/landmarks.xlsx')
    else:
        global_landmark_df.to_excel(f'{path_to_masks}/landmarks.xlsx')
        global_landmark_df[["landmark", "plant", "x", "y"]].sort_values(by="plant").to_excel(f'{path_to_masks}/landmarks.xlsx')
    cv2.imwrite(f'{path_to_masks}/image_mask.png', image)

def build_biased_graph_timeseries(
    branch_df,
    skeleton,
    head_length=50,
    angle_penalty=50.0,
    crop_padding=500,
    forbid_upward=True,
    upward_tolerance=10,
    upward_penalty=3.0
):
    """
    Build a graph from branch_df with edge costs that are biased towards vertical downward paths.

    Parameters:
        branch_df (pd.DataFrame)
            DataFrame with node-id-src, node-id-dst, image-coord-src-0, image-coord-src-1,
            image-coord-dst-0, image-coord-dst-1 columns.
        skeleton (np.ndarray)
            Binary skeleton image corresponding to the branch_df.
        head_length (int)
            Number of pixels from the source node to consider for angle calculation.
        angle_penalty (float)
            Multiplier for the angle-based cost penalty.
        crop_padding (int)
            Number of pixels to pad around the source coordinates.
        forbid_upward (bool)
            If True, forbids upward paths.
    Returns:
        nx.DiGraph: The resulting graph with biased edge costs.
    """
    G = nx.DiGraph()
    # Loop through the branches.
    for _, row in branch_df.iterrows():
        # Select required values from the row.
        src, dst = row["node-id-src"], row["node-id-dst"]
        y0, x0 = int(row["image-coord-src-0"]), int(row["image-coord-src-1"])
        y1, x1 = int(row["image-coord-dst-0"]), int(row["image-coord-dst-1"])
        G.add_node(src)
        G.add_node(dst)
        # Select crop area.
        ym0 = max(0, min(y0, y1) - crop_padding)
        ym1 = min(skeleton.shape[0], max(y0, y1) + crop_padding)
        xm0 = max(0, min(x0, x1) - crop_padding)
        xm1 = min(skeleton.shape[1], max(x0, x1) + crop_padding)

        sub = skeleton[ym0:ym1, xm0:xm1]
        try:
            # Calculate the path based on angle.
            path, _ = skimage.graph.route_through_array(
                1 - sub, (y0 - ym0, x0 - xm0), (y1 - ym0, x1 - xm0),
                fully_connected=True)
            if len(path) < 2:
                continue
            head = path[:min(head_length, len(path))]
            dy = head[-1][0] - head[0][0]
            dx = head[-1][1] - head[0][1]
            length = math.hypot(dx, dy)
            if (
                length == 0
                or forbid_upward
                and dy < -upward_tolerance
            ):
                continue
            # Calculate angle penalty and add edge to graph.
            angle_deg = math.degrees(math.acos(np.clip(dy / length, -1, 1)))
            cost = angle_penalty * (angle_deg / 180.0)
            G.add_edge(src, dst, cost=cost, dy=dy, dx=dx, angle_deg=angle_deg, length=length)
        except Exception:
            continue

    return G

def add_virtual_edges(G, coord_map, max_dist):
    """
    Add virtual edges between nearby skeleton tips in the graph.

    Parameters:
        G (nx.Graph)
            Graph with nodes corresponding to skeleton points.
        coord_map (dict)
            Mapping from node ID to (y, x) coordinates.
        max_dist (float)
            Maximum distance to connect tips with virtual edges.
    """
    # Find connected components and label them.
    components = list(nx.connected_components(G.to_undirected()))
    label_map = {n: i for i, comp in enumerate(components) for n in comp}
    tips_by_label = {}
    # loop through nodes.
    for node in G.nodes:
        # Check if node is a tip with a degree below or equal to 2.
        if G.degree(node) <= 2 and node in coord_map:
            # Add node to the corresponding label group.
            tips_by_label.setdefault(label_map[node], []).append(node)
    # Loop through tips and labels to find near tips.
    for l1, l2 in combinations(tips_by_label.keys(), 2):
        # Calculate best distance.
        best_dist, best_pair = np.inf, (None, None)
        for t1 in tips_by_label[l1]:
            for t2 in tips_by_label[l2]:
                d = np.linalg.norm(np.array(coord_map[t1]) - np.array(coord_map[t2]))
                if d < best_dist:
                    best_dist, best_pair = d, (t1, t2)
        # Skip if the best distance is greater than limited.
        if best_dist > max_dist:
            continue
        t1, t2 = best_pair
        y1, y2 = coord_map[t1][0], coord_map[t2][0]
        src, dst = (t1, t2) if y1 < y2 else (t2, t1)
        if not G.has_edge(src, dst):
            G.add_edge(src, dst, cost=best_dist, virtual=True)


def cumulative_angle(coords):
    """
    Calculate the cumulative angle deviation along a path defined by coordinates.

    Parameters:
        coords (list of tuples)
            List of (y, x) coordinates along the path.
    
    Returns:
        float: The cumulative angle deviation in radians.
    """
    total = 0
    # Loop through coordinates.
    for i in range(1, len(coords) - 1):
        # Calculate vectors and their norms.
        v1 = np.array([coords[i][1] - coords[i-1][1], coords[i][0] - coords[i-1][0]])
        v2 = np.array([coords[i+1][1] - coords[i][1], coords[i+1][0] - coords[i][0]])
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            continue
        total += math.acos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1))
    return total


def select_best_path_timeseries(G, start_node, coord_map, min_dy_ratio=0.8, target_tip_y=None):
    if start_node not in G:
        return []
    try:
        _, paths = nx.single_source_dijkstra(G, start_node, weight="cost")
    except nx.NetworkXNoPath:
        return []

    candidates = []
    for node, path in paths.items():
        if node == start_node or len(path) < 2:
            continue
        if len(path) != len(set(path)):
            continue
        coords = [coord_map[n] for n in path if n in coord_map]
        if len(coords) < 2:
            continue
        dy = coords[-1][0] - coords[0][0]
        if dy <= 0:
            continue

        angle_dev = cumulative_angle(coords)

        # How close does this path get to the target tip
        tip_y = coords[-1][0]
        tip_score = abs(tip_y - target_tip_y) if target_tip_y is not None else 0

        candidates.append({
            "path": path,
            "dy": dy,
            "angle_dev": angle_dev,
            "tip_score": tip_score
        })

    if not candidates:
        return []
    max_dy = max(c["dy"] for c in candidates)
    top = [c for c in candidates if c["dy"] >= max_dy * min_dy_ratio]

    if target_tip_y is not None:
        # Among long enough paths, prefer ones closest to target tip y
        # then break ties with angle deviation
        return min(top, key=lambda c: (c["tip_score"], c["angle_dev"]))["path"]
    else:
        return min(top, key=lambda c: c["angle_dev"])["path"]

def get_closest(branch_df, row, threshold=45):
    """
    Get the closest branch to the given row based on proximity and angle.

    Parameters:
        branch_df (pd.DataFrame)
            DataFrame with node-id-src, node-id-dst, image-coord-src-0, image-coord-src-1,
            image-coord-dst-0, image-coord-dst-1, and root_type columns.
        row (pd.Series)
            A row from branch_df representing the current branch to compare against.
        threshold (float)
            Maximum distance to consider for a branch to be a candidate.

    Returns:
        pd.Series or None
            The closest branch as a Series, or None if no suitable candidate is found.
    """
    # Extract coordinates and angle from the input row.
    px, py, angle = row["image-coord-dst-1"], row["image-coord-dst-0"], row["angle"]
    dist = np.sqrt(
        (branch_df["coord-src-1"] - px) ** 2 +
        (branch_df["coord-src-0"] - py) ** 2
    )
    # Select candidates.
    candidates = branch_df[
        (dist <= threshold) &
        (branch_df["node-id-src"] != row["node-id-src"]) &
        (branch_df["node-id-dst"] != row["node-id-dst"]) &
        (branch_df["root_type"] != "Primary")
    ]
    if candidates.empty:
        return None
    # Return the candidate with the closest angle.
    return candidates.iloc[(candidates["angle"] - angle).abs().argsort()[:1]].iloc[0]


def follow_lateral_path(branch_df):
    """
    Follow paths from primary roots to assign lateral roots based on proximity and angle.

    Parameters:
        branch_df (pd.DataFrame)
            DataFrame with "node-id-src", "node-id-dst", "image-coord-src-0", "image-coord-src-1",

    Returns:
        pd.DataFrame
            Updated branch_df with lateral roots assigned.
    """
    # Loop through assigned plants and follow paths to assign laterals.
    assigned = branch_df[branch_df["plant"] != "None"]
    # Loop through plants.
    for plant in sorted(assigned["plant"].unique()):
        if plant == "None" or (isinstance(plant, float) and np.isnan(plant)):
            continue
        plant = int(plant)
        primary_rows = branch_df[
            (branch_df["plant"] == plant) &
            (branch_df["root_type"] == "Primary")
        ]
        root_id = 0
        # Loop through primary rows.
        for _, row in primary_rows.iterrows():
            steps, following = 0, True
            # follow the paths.
            while following:
                closest = get_closest(branch_df, row if steps == 0 else closest)
                if closest is None or steps > 10:
                    break
                idx = closest.name
                branch_df.at[idx, "plant"] = plant
                branch_df.at[idx, "root_type"] = "Lateral"
                branch_df.at[idx, "root_id"] = f"Lateral_{root_id}"
                closest = branch_df.loc[idx]
                steps += 1
            root_id += 1
    return branch_df


def get_masks(branch_df, skeleton_ob, original_shape, plant_idx=None):
    """
    Generate primary and lateral masks from branch assignments.

    Parameters:
        branch_df: Output from segment_roots.
        skeleton_ob: skan Skeleton object.
        original_shape: (h, w) of the original image.
        plant_idx: If provided, only generate masks for that plant.

    Returns:
        primary_mask, lateral_mask: Binary numpy arrays.
    """
    primary_mask = np.zeros(original_shape, dtype=np.uint8)
    lateral_mask = np.zeros(original_shape, dtype=np.uint8)

    for idx, row in branch_df.iterrows():
        if plant_idx is not None and row["plant"] != plant_idx:
            continue
        coords = skeleton_ob.path_coordinates(idx)
        for y, x in coords:
            if 0 <= y < original_shape[0] and 0 <= x < original_shape[1]:
                if row["root_type"] == "Primary":
                    primary_mask[y, x] = 1
                elif row["root_type"] == "Lateral":
                    lateral_mask[y, x] = 1

    return primary_mask, lateral_mask