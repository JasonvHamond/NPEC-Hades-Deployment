### IMPORTANT ###
# Codes in this file are based on the original pipeline codes created by BUas students from last year:
# https://github.com/JasonvHamond/NPEC-Hades-Deployment/blob/main/npec_hades_deployment/src/inference/roots_segmentation.py
# Function-specific references to original author can be found there,
# The codes in this file are rewritten to ensure further originality.

import math
import re
import warnings
import numpy as np
import cv2
import networkx as nx
import pandas as pd
import skimage.morphology
import skimage.graph
import skimage.measure
from itertools import combinations
from skan import Skeleton, summarize
from skimage.draw import line

def overlaps_any(proposed, idx, final_boxes):
    """
    Check if the proposed box overlaps with any of the final boxes except the one at idx.
    Parameters:
        proposed (list):
            [ymin, ymax, xmin, xmax] of the proposed box.
        idx (int):
            Index of the box in final_boxes to ignore during overlap check.
        final_boxes (list of lists):
            List of [ymin, ymax, xmin, xmax] for existing boxes.
    """
    py0, py1, px0, px1 = proposed
    for j, (oy0, oy1, ox0, ox1) in enumerate(final_boxes):
        if j == idx:
            continue
        if not (px1 <= ox0 or px0 >= ox1 or py1 <= oy0 or py0 >= oy1):
            return True
    return False

def get_plant_bboxes(
    root_mask,
    expected_centers,
    initial_box_halfsize_x=200,
    initial_box_halfsize_y=200,
    expansion_step=1,
    max_empty_expansions=100
):
    """
    Grow bounding boxes from seed centers until no more root pixels are found.
    
    Parameters:
        root_mask (np.ndarray)
            Binary numpy array of the root mask.
        expected_centers (list of tuples)
            List of (cx, cy, id) tuples for expected plant seed positions.
        initial_box_halfsize_x (int)
            Initial half-width of the bounding box around each seed center.
        initial_box_halfsize_y (int)
            Initial half-height of the bounding box around each seed center.
        expansion_step (int)
            Number of pixels to expand the bounding box in each step.
        max_empty_expansions (int)
            Maximum number of consecutive expansions allowed without finding new root pixels before 
            stopping expansion in that direction.
    
    Returns:
        final_boxes (list of lists)
            List of [ymin, ymax, xmin, xmax] bounding boxes for each expected center.
    """
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.dilate(root_mask.astype(np.uint8), kernel, iterations=1)
    binary = (binary > 0).astype(np.uint8)
    h, w = binary.shape

    final_boxes = []
    empty_counters = []
    can_expand = []
    # Loop through the expected centers and initialize bounding boxes.
    for cx, cy, _ in expected_centers:
        ymin = max(cy - initial_box_halfsize_y, 0)
        ymax = min(cy + initial_box_halfsize_y, h)
        xmin = max(cx - initial_box_halfsize_x, 0)
        xmax = min(cx + initial_box_halfsize_x, w)
        # Select region of the binary mask.
        region = binary[ymin:ymax, xmin:xmax]
        ys, xs = np.where(region > 0)
        # Check if any root pixels are found in initial box.
        if len(xs) > 0:
            # Adjust box to fit to found root pixels.
            ymin = max(ymin + int(ys.min()) - 1, 0)
            ymax = min(ymin + int(ys.max()) + 2, h)
            xmin = max(xmin + int(xs.min()) - 1, 0)
            xmax = min(xmin + int(xs.max()) + 2, w)
        # Add the initial box to the final boxes list.
        final_boxes.append([ymin, ymax, xmin, xmax])
        empty_counters.append({"up": 0, "down": 0, "left": 0, "right": 0})
        can_expand.append({"up": True, "down": True, "left": True, "right": True})
    # Expand boxes as far as possible..
    still_expanding = True
    while still_expanding:
        still_expanding = False
        # Loop through the expected centers to expand their boxes.
        for i in range(len(expected_centers)):
            ymin, ymax, xmin, xmax = final_boxes[i]
            # Loop through the four possible directions.
            for direction in ["up", "down", "left", "right"]:
                if not can_expand[i][direction]:
                    continue
                # Change the box coordinates based on the direction.
                if direction == "up":
                    ny = max(ymin - expansion_step, 0)
                    region = binary[ny:ymin, xmin:xmax]
                    proposed = [ny, ymax, xmin, xmax]
                elif direction == "down":
                    ny = min(ymax + expansion_step, h)
                    region = binary[ymax:ny, xmin:xmax]
                    proposed = [ymin, ny, xmin, xmax]
                elif direction == "left":
                    nx_ = max(xmin - expansion_step, 0)
                    region = binary[ymin:ymax, nx_:xmin]
                    proposed = [ymin, ymax, nx_, xmax]
                else:
                    nx_ = min(xmax + expansion_step, w)
                    region = binary[ymin:ymax, xmax:nx_]
                    proposed = [ymin, ymax, xmin, nx_]
                # Check if the proposed box overlaps with any of the existing boxes.
                if overlaps_any(proposed, i, final_boxes):
                    can_expand[i][direction] = False
                    continue
                # Update box if root pixels are found.
                if np.sum(region) > 0:
                    final_boxes[i] = proposed
                    empty_counters[i][direction] = 0
                    still_expanding = True
                # If none are found, increase the empty counter.
                else:
                    empty_counters[i][direction] += 1
                    # Check if the empty counter has reached the limit.
                    if empty_counters[i][direction] >= max_empty_expansions:
                        can_expand[i][direction] = False
    return final_boxes


def build_biased_graph(
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


def select_best_path(G, start_node, coord_map, min_dy_ratio=0.8, target_tip_y=None):
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


def find_primary_path(subset_branch, full_branch, G, coord_map, min_coverage=0.8, known_start_coord=None):
    """
    Search for the best path fitting to be a primary root.

    Parameters:
        subset_branch (pd.DataFrame)
            Subset of branch_df for the current plant.
        full_branch (pd.DataFrame)
            Full branch_df for all plants.
        G (nx.DiGraph)
            Graph with edge costs.
        coord_map (dict)
            Mapping from node ID to (y, x) coordinates.
        min_coverage (float)
            Minimum coverage ratio of the path's vertical displacement to the total height of the plant to be
            considered valid.
    
    Returns:
        list
            The best path as a list of node IDs, or an empty list if no valid path is found.
    """
    # Calculate total height of the plant.
    y_vals = subset_branch["image-coord-src-0"].tolist() + subset_branch["image-coord-dst-0"].tolist()
    total_height = np.ptp(y_vals)
    target_tip_y = max(y_vals)
    # Create mapping from node ID to y-coordinate for both subset and full branches.
    id_to_y = {**subset_branch.set_index("node-id-src")["image-coord-src-0"].to_dict(),
               **subset_branch.set_index("node-id-dst")["image-coord-dst-0"].to_dict()}
    id_to_y_full = {**full_branch.set_index("node-id-src")["image-coord-src-0"].to_dict(),
                    **full_branch.set_index("node-id-dst")["image-coord-dst-0"].to_dict()}

    all_src = set(subset_branch["node-id-src"])
    all_dst = set(subset_branch["node-id-dst"])
    root_nodes = all_src - all_dst
    candidates = root_nodes if root_nodes else all_src
    if known_start_coord is not None:
        # Pick candidate closest to the known start position
        def dist_to_known(n):
            y, x = id_to_y.get(n, 99999), coord_map.get(n, (0, 99999))[1]
            return np.linalg.norm(np.array([y, x]) - np.array(known_start_coord))
        sorted_candidates = sorted(candidates, key=dist_to_known)
    else:
        # Fall back to topmost node
        sorted_candidates = sorted(candidates, key=lambda n: id_to_y.get(n, 99999))

    best_path, best_coverage = [], -1
    for node in sorted_candidates:
        # path = select_best_path(G, node, coord_map)
        path = select_best_path(G, node, coord_map, target_tip_y=target_tip_y)
        if not path or len(path) < 2:
            continue
        if len(path) != len(set(path)):
            continue
        height = id_to_y_full.get(path[-1], 0) - id_to_y.get(path[0], 0)
        coverage = height / total_height if total_height > 0 else 0
        if coverage > best_coverage:
            best_path, best_coverage = path, coverage
        if best_coverage >= min_coverage:
            break

    return best_path


def reconnect_skeleton(skeleton, max_dist):
    """
    Reconnect disconnected components in a skeleton image.
    This function is originally created by Wesley van Gaalen,
    a BUas student from last year.

    Parameters:
        skeleton (np.ndarray)
            Binary skeleton image.
        max_dist (float)
            Maximum distance to connect components.

    Returns:
        np.ndarray
            Reconnected skeleton image.
    """
    # dilate skeleton to make connections.
    expand = skimage.morphology.dilation(skeleton, skimage.morphology.square(1))
    labeled = skimage.measure.label(expand)
    props = skimage.measure.regionprops(labeled)
    # Loop through regions and nearby endpoints.
    for i, r1 in enumerate(props):
        # Skip if region is too small to be a valid root segment.
        best_score, best_coords = float("inf"), None
        y1_mean = np.mean(r1.coords[:, 0])
        # Loop through other regions to find nearby endpoints.
        for j, r2 in enumerate(props):
            if i == j:
                continue
            if np.mean(r2.coords[:, 0]) < y1_mean:
                continue
            # Calculate min. distances.
            dists = np.sqrt(((r1.coords[:, None] - r2.coords) ** 2).sum(-1))
            d = np.min(dists)
            # Update best score and coordinates if distance fits threshold.
            if d < max_dist and d < best_score:
                best_score = d
                i1, i2 = np.unravel_index(np.argmin(dists), dists.shape)
                best_coords = (r1.coords[i1], r2.coords[i2])
        # If a good cordinate pair is found, connect them.
        if best_coords:
            (y0, x0), (y1, x1) = best_coords
            rr, cc = line(y0, x0, y1, x1)
            skeleton[rr, cc] = 1
    return skeleton


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


def assign_laterals(branch_df, skeleton_ob, plant_bboxes):
    """
    Assign lateral roots based on proximity to primary roots and plant bounding boxes.

    Parameters:
        branch_df (pd.DataFrame)
            DataFrame with "node-id-src", "node-id-dst", "image-coord-src-0", "image-coord-src-1",
            "image-coord-dst-0", "image-coord-dst-1", and "root_type" columns.
        skeleton_ob (Skeleton)
            skan Skeleton object corresponding to the branch_df.
        plant_bboxes (list of lists)
            List of [ymin, ymax, xmin, xmax] bounding boxes for each plant.

    Returns:
        pd.DataFrame
            Updated branch_df with lateral roots assigned.
    """
    # Create copy of branch df so original is not modified.
    df_copy = branch_df.copy()
    # Calculate angles and add as column.
    df_copy["angle"] = df_copy.apply(
        lambda r: math.degrees(math.atan2(
            r["image-coord-dst-0"] - r["image-coord-src-0"],
            r["image-coord-dst-1"] - r["image-coord-src-1"])), axis=1)
    # Follow paths from primary roots to assign laterals.
    df_copy = follow_lateral_path(df_copy)

    counter_start = 0
    existing = df_copy["root_id"].dropna().astype(str)
    nums = [int(re.findall(r"\d+", r)[0]) for r in existing if re.findall(r"\d+", r)]
    if nums:
        counter_start = max(nums) + 1
    root_id_counter = counter_start
    # Loop through branches.
    for idx, row in branch_df.iterrows():
        # Skip if already assigned as primary or lateral.
        if row["root_type"] == "Primary" or pd.notna(row.get("root_id")):
            continue
        # Get coordinates.
        coords = skeleton_ob.path_coordinates(idx)
        if len(coords) == 0:
            continue
        y0, x0 = coords[0]
        # Loop through plant bounding boxes
        for plant_idx, (ymin, ymax, xmin, xmax) in enumerate(plant_bboxes):
            # find if branch is within any box.
            if ymin <= y0 <= ymax and xmin <= x0 <= xmax:
                branch_df.at[idx, "plant"] = plant_idx
                branch_df.at[idx, "root_type"] = "Lateral"
                branch_df.at[idx, "root_id"] = f"Lateral_{root_id_counter}"
                root_id_counter += 1
                break

    return branch_df


def segment_roots(root_mask, expected_centers, reconnect_max_dist=20.0, known_start_coords=None):
    """
    Takes a binary root mask, returns branch_df with primary/lateral
    assignments, the Skeleton object, and plant bounding boxes.

    Parameters:
        root_mask (np.ndarray)
            Binary numpy array from model_predict.
        expected_centers (list)
            List of (cx, cy, id) tuples for plant seed positions.
        reconnect_max_dist (float)
            Max distance to add virtual edges between skeleton components.

    Returns:
        branch_df (pd.DataFrame)
            DataFrame with root_type (Primary/Lateral/None) and plant.
        skeleton_ob (skeleton.Skeleton) 
            skan Skeleton object.
        plant_bboxes (list)
            List of [ymin, ymax, xmin, xmax] per plant.
    """
    # Prepare mask.
    mask = root_mask.astype(np.uint8)
    for _ in range(3):
        mask = skimage.morphology.dilation(mask, skimage.morphology.square(3))

    # Skeletonize.
    skeleton = skimage.morphology.skeletonize(mask)

    # Reconnect broken skeleton segments.
    skeleton = reconnect_skeleton(skeleton, reconnect_max_dist)

    if len(np.unique(skeleton)) == 1:
        warnings.warn("Skeleton is empty.")
        return None, None, None
    # Summarize skeleton to get branch_df.
    skeleton_ob = Skeleton(skeleton)
    branch_df = summarize(skeleton_ob)
    branch_df["root_type"] = "None"
    branch_df["plant"] = "None"
    branch_df["root_id"] = np.nan
    branch_df["root_id"] = branch_df["root_id"].astype(object)

    # Build graph.
    G = build_biased_graph(branch_df, skeleton)
    # Create coordinate mapping for graph nodes.
    coord_map = {row["node-id-src"]: (row["image-coord-src-0"], row["image-coord-src-1"])
                 for _, row in branch_df.iterrows()}
    coord_map.update({row["node-id-dst"]: (row["image-coord-dst-0"], row["image-coord-dst-1"])
                      for _, row in branch_df.iterrows()})

    # Add virtual edges between nearby skeleton tips.
    add_virtual_edges(G, coord_map, reconnect_max_dist)

    # Get plant bounding boxes.
    plant_bboxes = get_plant_bboxes(root_mask, expected_centers)

    # Find primary root.
    for plant_num, bbox in enumerate(plant_bboxes):
        ymin, ymax, xmin, xmax = bbox
        # Get coordinates of skeleton points within the bounding box.
        coords_in_box = set(zip(*np.where(skeleton[ymin:ymax, xmin:xmax] > 0)))
        coords_in_box = {(y + ymin, x + xmin) for y, x in coords_in_box}

        subset = branch_df[branch_df.apply(
            lambda r: (
                r["image-coord-src-0"],
                r["image-coord-src-1"]
            ) in coords_in_box, axis=1
        )].copy()

        if subset.empty:
            continue
        known_coord = known_start_coords.get(plant_num) if known_start_coords else None
        path = find_primary_path(subset, branch_df, G, coord_map, known_start_coord=known_coord)

        # path = find_primary_path(subset, branch_df, G, coord_map)
        if not path:
            continue

        primary_pairs = set(zip(path[:-1], path[1:]))
        for idx, row in branch_df.iterrows():
            if (row["node-id-src"], row["node-id-dst"]) in primary_pairs:
                branch_df.at[idx, "root_type"] = "Primary"
                branch_df.at[idx, "plant"] = plant_num
                branch_df.at[idx, "root_id"] = "Primary"

        pair_to_indices = {}
        for idx, row in branch_df[branch_df["root_type"] == "Primary"].iterrows():
            pair = tuple(sorted((row["node-id-src"], row["node-id-dst"])))
            pair_to_indices.setdefault(pair, []).append(idx)
        for indices in pair_to_indices.values():
            if len(indices) > 1:
                best_idx = min(indices, key=lambda i: branch_df.loc[i, "branch-distance"])
                for idx in indices:
                    if idx != best_idx:
                        branch_df.at[idx, "root_type"] = "None"

    # Assign lateral roots.
    branch_df = assign_laterals(branch_df, skeleton_ob, plant_bboxes)

    return branch_df, skeleton_ob, plant_bboxes


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