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

import inference.roots_segmentation as post

def interpolate_path(coords_prev, coords_next, prev_weight=0.9):
    """
    Interpolate between two paths, giving more weight to the previous path.
    This will ensure that, if no root exists in the current image,
    the previous path will be used as basis and continued based on future growth.

    Paramters:
    - coords_prev (np.array):
        Coordinates of previous path.
    - coords_next (np.array):
        Coordinates of next path.
    - prev_weight (float):
        Weight for previous path in interpolation (between 0 and 1).
    
    Returns:
    - interp_coords (np.array):
        Interpolated coordinates.
    """
    # Normalise paths to [0, 1] for interpolation.
    t_prev = np.linspace(0, 1, len(coords_prev))
    t_next = np.linspace(0, 1, len(coords_next))

    # Find tip positions.
    prev_tip_y = coords_prev[np.argmax(coords_prev[:, 0]), 0]
    next_tip_y = coords_next[np.argmax(coords_next[:, 0]), 0]

    # Calculate where the tip should end.
    interp_tip_y = (prev_tip_y + next_tip_y) / 2 if next_tip_y > prev_tip_y else prev_tip_y

    # Check how far the tip is past previous tip.
    prev_max_y = coords_prev[:, 0].max()
    
    # Resample shared path.
    n_shared = len(coords_prev)
    t_shared = np.linspace(0, 1, n_shared)
    # Interpolate both paths to shared t.
    shared_y = np.interp(
        t_shared,
        t_prev,
        coords_prev[:, 0]
        ) * prev_weight + np.interp(
            t_shared,
            t_next,
            coords_next[:, 0]
        ) * (1 - prev_weight)
    shared_x = np.interp(
        t_shared,
        t_prev,
        coords_prev[:, 1]
    ) * prev_weight + np.interp(
        t_shared,
        t_next,
        coords_next[:, 1]
    ) * (1 - prev_weight)

    # Find points in next path past previous tip.
    next_beyond = coords_next[coords_next[:, 0] > prev_max_y]

    if len(next_beyond) > 0 and next_tip_y > prev_tip_y:
        # Start extension from end of shared path.
        offset_y = shared_y[-1] - next_beyond[0, 0]
        offset_x = shared_x[-1] - next_beyond[0, 1]
        extension_y = next_beyond[:, 0] + offset_y
        extension_x = next_beyond[:, 1] + offset_x

        # Include only extension points below interpolated position.
        valid = extension_y <= interp_tip_y
        extension_y = extension_y[valid]
        extension_x = extension_x[valid]
        # Concatenate shared and extension parts.
        if len(extension_y) > 0:
            interp_y = np.concatenate([shared_y, extension_y])
            interp_x = np.concatenate([shared_x, extension_x])
            return np.column_stack([interp_y, interp_x])

    # No extension needed. Just return shared part up to interpolated tip.
    valid = shared_y <= interp_tip_y
    return np.column_stack([shared_y[valid], shared_x[valid]])


def path_to_mask(coords, shape, thickness=1):
    """
    Convert a coordinate path into binary mask of given shape and thickness.

    Parameters:
    - coords (np.array):
        Array containing coordinates of the path.
    - shape (tuple):
        Shape of the output mask (height, width).
    - thickness (int):
        Pixel thickness of the path in the mask.
    
    Returns:
    - mask (np.array):
        Binary mask with the path drawn.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    coords_int = coords.astype(int)

    for i in range(len(coords_int) - 1):
        rr, cc = skimage.draw.line(
            coords_int[i, 0], coords_int[i, 1],
            coords_int[i+1, 0], coords_int[i+1, 1]
        )
        valid = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
        mask[rr[valid], cc[valid]] = 1
    if thickness > 1:
        mask = skimage.morphology.dilation(
            mask, skimage.morphology.disk(thickness)
        )
    return mask


def get_primary_coords(branch_df, skeleton_ob, plant_idx):
    """
    Get the coordinates of the primary root path.

    Parameters:
    - branch_df (pd.DataFrame):
        DataFrame containing branch information.
    - skeleton_ob (Skeleton):
        Skeleton object containing path information.
    - plant_idx (int):
        Index of the plant to get coordinates for.
        
    Returns:
    - coords (np.array):
        Array of coordinates for the primary root path. Empty if no primary path exists.
    """
    # Get the coordinates for the current Primary path.
    subset = branch_df[
        (branch_df["root_type"] == "Primary") &
        (branch_df["plant"] == plant_idx)
    ]
    coords = []
    # Loop through the subset, collect coordinates.
    for idx in subset.index:
        coords.extend(skeleton_ob.path_coordinates(idx).tolist())
    # Return the nump array if coords exists.
    return np.array(coords) if coords else np.empty((0, 2))


def path_similarity(coords_prev, coords_curr, max_dist=50):
    """
    Calculate similarity between two paths based on average distance between points.

    Parameters:
    - coords_prev (np.array):
        Coordinates of previous path.
    - coords_curr (np.array):
        Coordinates of current path.
    - max_dist (float):
        Maximum distance to consider points as matching (in pixels).

    Returns:
    - similarity (float):
        Average fraction of points in current path within max_dist.
    """
    # Check if the length of coordinates is zero to avoid future errors.
    if len(coords_prev) == 0 or len(coords_curr) == 0:
        return 0.0
    # Build tree for previous coordinates.
    tree = KDTree(coords_prev)
    # Check distances between current and previous coordinates.
    dists, _ = tree.query(coords_curr)
    return float(np.mean(dists <= max_dist))


def tip_displacement(coords_prev, coords_curr):
    """
    Calculate the displacement of the tip between two paths.

    Parameters:
    - coords_prev (np.array):
        Coordinates of previous path.
    - coords_curr (np.array):
        Coordinates of current path.

    Returns:
    - displacement (float):
        Euclidean distance between the tips of the two paths.
    """
    if len(coords_prev) == 0 or len(coords_curr) == 0:
        return np.inf
    tip_prev = coords_prev[np.argmax(coords_prev[:, 0])]
    tip_curr = coords_curr[np.argmax(coords_curr[:, 0])]
    return float(np.linalg.norm(tip_curr - tip_prev))

def find_consistent_growth_path(
    predictions,
    filenames,
    coords_per_plant,
    plant_idx,
    t
):
    """
    Find the most consistent growth path for a plant at time t based on the previous time step.
    This function checks for alternative paths in the current prediction and selects
    the most similar to the previous path, ensuring consistency between temporal steps.

    Parameters:
    - predictions (dict):
        Dictionary containing predictions for each file, including branch information and skeletons.
    - filenames (list):
        List of filenames corresponding to the predictions, ordered by time.
    - coords_per_plant (dict):
        Dictionary containing coordinates of the primary path for each plant at each time step.
    - plant_idx (int):
        Index of the plant to correct.
    - t (int):
        Current time step index for which to find the consistent path.

    Returns:
    - predictions (dict):
        Updated predictions with the most consistent path selected for the specified plant and time step.
    - coords_per_plant (dict):
        Updated coordinates per plant with the new primary path coordinates for the specified plant and time step.
    """
    # Collect important data for which files are tackled.
    filename_curr = filenames[t]

    # Collect previous coordinates.
    prev_coords = coords_per_plant[plant_idx][t-1]

    if len(prev_coords) == 0:
        print("No previous coords available, skipping.")
        return predictions, coords_per_plant

    prev_tree = KDTree(prev_coords)

    # Get all required data.
    branch_df = predictions[filename_curr]["branches"].copy()

    skeleton_ob = predictions[filename_curr]["skeleton"]
    plant_branches = branch_df[branch_df["plant"] == plant_idx]

    # Measure mean distance to previous path.
    segment_scores = {}
    for idx in plant_branches.index:
        coords = skeleton_ob.path_coordinates(idx)
        dists, _ = prev_tree.query(coords)
        segment_scores[idx] = float(np.mean(dists))
    # Find best alternative segment.
    sorted_segments = sorted(segment_scores.items(), key=lambda x: x[1])
    _, best_score = sorted_segments[0]
    
    current_primary_indices = branch_df[
        (branch_df["root_type"] == "Primary") &
        (branch_df["plant"] == plant_idx)
    ].index
    if len(current_primary_indices) == 0:
        current_primary_score = float('inf')
    else:
        scores = [segment_scores[i] for i in current_primary_indices if i in segment_scores]
        current_primary_score = np.mean(scores) if scores else float('inf')

    # Get the original primary coordinates for future comparisons.
    orig_primary_coords = get_primary_coords(branch_df, skeleton_ob, plant_idx)
    orig_start_y = orig_primary_coords[:, 0].min() if len(orig_primary_coords) > 0 else None
    # Check for at least 20% improvement of the new scores.
    if best_score < current_primary_score * 0.8:
        # Save current primary root as lateral root.
        # This will help with primary root selection without needing to worry about -
        # missing primary roots
        branch_df.loc[current_primary_indices, "root_type"] = "Lateral"
        plant_df = branch_df[branch_df["plant"] == plant_idx]

        # Only consider true root nodes.
        all_src = set(plant_df["node-id-src"])
        all_dst = set(plant_df["node-id-dst"])
        root_nodes = all_src - all_dst

        start_node_scores = {}
        # Loop through root nodes.
        for idx, row in branch_df[branch_df["plant"] == plant_idx].iterrows():
            node = row["node-id-src"]
            if node not in root_nodes:
                continue
            #Select the y coordinate to help scoring.
            y = row["image-coord-src-0"]
            coord = np.array([y, row["image-coord-src-1"]])
            # Calculate distance to previous path.
            dist_to_prev, _ = prev_tree.query(coord)
            # Reward high nodes which are closest to previous path.
            start_node_scores[node] = dist_to_prev + y * 0.01
        # When no root nodes are found, try all nodes.
        if not start_node_scores:
            for idx, row in plant_df.iterrows():
                node = row["node-id-src"]
                y = row["image-coord-src-0"]
                coord = np.array([y, row["image-coord-src-1"]])
                dist_to_prev, _ = prev_tree.query(coord)
                start_node_scores[node] = dist_to_prev + y * 0.01
        # Select best node on lowest score.
        best_start_node = min(start_node_scores, key=start_node_scores.get)
        # create coordinate map.
        coord_map = {
            row["node-id-src"]: (
                row["image-coord-src-0"],
                row["image-coord-src-1"]
            ) for _, row in branch_df.iterrows()
        }
        # update map.
        coord_map.update({
            row["node-id-dst"]: (
                row["image-coord-dst-0"],
                row["image-coord-dst-1"]
            ) for _, row in branch_df.iterrows()
        })
        plant_df = branch_df[branch_df["plant"] == plant_idx]
        all_y_vals = (
            plant_df["image-coord-src-0"].tolist() + 
            plant_df["image-coord-dst-0"].tolist()
        )
        target_tip_y = max(all_y_vals) if all_y_vals else None
        
        G = post.build_biased_graph_timeseries(branch_df, skeleton_ob.skeleton_image)
        post.add_virtual_edges(G, coord_map, max_dist=200)
        for u, v, data in G.edges(data=True):
            if u in coord_map:
                coord = np.array(coord_map[u])
                dist_to_prev, _ = prev_tree.query(coord)
                # Add distance to previous path as extra cost.
                data["cost"] = data.get("cost", 0) + dist_to_prev * 1.0

        # Collect all possible paths.
        _, all_paths = nx.single_source_dijkstra(G, best_start_node, weight="cost")
        max_dy = max(
            (coord_map[n][0] - coord_map[best_start_node][0])
            for n in all_paths if n in coord_map
        )

        candidates = []
        # Collect candidates for the best path.
        for end_node, path in all_paths.items():
            if len(path) < 2 or end_node not in coord_map:
                continue
            if len(path) != len(set(path)):
                continue
            # Try making sure the path is calculated more downward.
            dy = coord_map[end_node][0] - coord_map[best_start_node][0]
            if dy < max_dy * 0.9:
                continue
            # Calculate similarity and collect it together with the path.
            coords = np.array([coord_map[n] for n in path if n in coord_map])
            dists, _ = prev_tree.query(coords)
            similarity = float(np.mean(dists <= 50))
            prev_tip = prev_coords[np.argmax(prev_coords[:, 0])]
            prev_start = prev_coords[0]
            prev_dir = prev_tip - prev_start
            prev_dir = prev_dir / (np.linalg.norm(prev_dir) + 1e-6)

            curr_dir = coords[-1] - coords[0]
            curr_dir = curr_dir / (np.linalg.norm(curr_dir) + 1e-6)

            # Dot product — penalize paths going in a very different direction to previous
            direction_alignment = float(np.dot(prev_dir, curr_dir))
            score = similarity * 0.6 + max(0, direction_alignment) * 0.4
            candidates.append((path, score))

            # candidates.append((path, similarity))
        # Select the highest possible path if there are candidates
        if candidates:
            path = max(candidates, key=lambda x: x[1])[0]
        else:
            # Just run the select best paths function when there are no candidates
            # path = post.select_best_path(G, best_start_node, coord_map)
            path = post.select_best_path(G, best_start_node, coord_map, target_tip_y=target_tip_y)

        # If path calculation is successful, continue with updating.
        if path:
            # Get the new path coordinates for further comparisons and checks.
            new_path_coords = np.array([coord_map[n] for n in path if n in coord_map])
            new_start_y = new_path_coords[:, 0].min() if len(new_path_coords) > 0 else None
            # If start node is lower than the original, recalculate new path using original start node.
            if orig_start_y is not None and new_start_y is not None and new_start_y - orig_start_y > 50:
                print(f"Corrected path starts lower than original, recalculating from original start.")
                # Find the candidate node closest to the original start y.
                plant_df = branch_df[branch_df["plant"] == plant_idx]
                all_src = set(plant_df["node-id-src"])
                all_dst = set(plant_df["node-id-dst"])
                root_nodes = all_src - all_dst
                candidates = root_nodes if root_nodes else all_src
                # Select the candidate closest to the original start y.
                best_start = min(candidates, key=lambda n: abs(coord_map[n][0] - orig_start_y) if n in coord_map else 99999)
                path = post.select_best_path(G, best_start, coord_map, target_tip_y=target_tip_y)

            # Check again if path exists due to possible recalculation.
            if path:
                primary_pairs = set(zip(path[:-1], path[1:]))
            for idx, row in branch_df.iterrows():
                if (row["node-id-src"], row["node-id-dst"]) in primary_pairs:
                    branch_df.at[idx, "root_type"] = "Primary"
                    branch_df.at[idx, "plant"] = plant_idx
        # Store masks per plant.
        for p_idx in branch_df["plant"].unique():
            if p_idx == "None" or (isinstance(p_idx, float) and np.isnan(p_idx)):
                continue
            p_idx = int(p_idx)
            p_primary, p_lateral = post.get_masks(
                branch_df, skeleton_ob, 
                predictions[filename_curr]["roots"].shape,
                plant_idx=p_idx
            )
            predictions[filename_curr][f"primary_plant{p_idx}"] = p_primary
            predictions[filename_curr][f"lateral_plant{p_idx}"] = p_lateral
        # Reset lateral roots for this plant.
        plant_lateral_mask = (branch_df["plant"] == plant_idx) & (branch_df["root_type"] == "Lateral")
        branch_df.loc[plant_lateral_mask, "root_type"] = "None"
        branch_df.loc[plant_lateral_mask, "plant"] = "None"
        branch_df.loc[plant_lateral_mask, "root_id"] = np.nan

        # Recalculate lateral roots.
        branch_df["angle"] = branch_df.apply(
            lambda r: math.degrees(math.atan2(
                r["image-coord-dst-0"] - r["image-coord-src-0"],
                r["image-coord-dst-1"] - r["image-coord-src-1"])), axis=1)
        branch_df = post.follow_lateral_path(branch_df)


        # Update masks and predictions.
        primary_mask, lateral_mask = post.get_masks(
            branch_df, skeleton_ob, predictions[filename_curr]["roots"].shape
        )
        # Update the predictions with the new branch_df and masks.
        predictions[filename_curr]["branches"] = branch_df
        predictions[filename_curr]["primary"] = primary_mask
        predictions[filename_curr]["lateral"] = lateral_mask

        new_coords = get_primary_coords(branch_df, skeleton_ob, plant_idx)
        coords_per_plant[plant_idx][t] = new_coords

        return predictions, coords_per_plant
    else:
        print("No alternative found.")
        return predictions, coords_per_plant
