import pandas as pd
from pathlib import Path
import re
import numpy as np
from PIL import Image
import pillow_jxl
import cv2
from patchify import patchify


def load_image(image_path: str) -> np.ndarray:
    """
    Load a JXL image from a given path.

    Author: Jason van Hamond

    Parameters:
    image_path (str): Path of the JXL image to load.

    Returns:
    np.ndarray: Loaded image.
    """
    image = Image.open(image_path)
    image = np.array(image)
    # Convert from RGB to BGR if necessary for CV2 compatibility.
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image =  image

    if len(image.shape) != 2:
        # Finally convert to grayscale.
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return image


def load_dataset(data_root_path, petri_dish_n = None):
    data_root = Path(data_root_path)
    records = []
    petri_dishes = [d for d in data_root.iterdir() 
                    if d.is_dir() and d.name.startswith("petri_dish")]
    for dish in petri_dishes:
        if petri_dish_n and dish.name != f"petri_dish{petri_dish_n}":
            continue
        annotations_dir = dish / "Annotations"
        images_dir = dish / "Images"
        if not annotations_dir.exists() or not images_dir.exists():
            print(f"Missing directories in {dish.name}")
            continue
        image_lookup = {}
        for image_file in list(images_dir.glob("*.png")) + list(images_dir.glob("*.jxl")):
            match = re.match(r"\d+_(.+?)_", image_file.name)
            if match:
                image_id = match.group(1)
                image_lookup[image_id] = image_file
        
        # Process annotations.
        for ann_file in annotations_dir.glob("*.tif"):
            match = re.match(r"(.*?)_plant(\d+)_(primary|lateral)\.tif", ann_file.name)
            if match:
                image_id, plant_num, root_type = match.groups()

                if image_id in image_lookup:
                    records.append({
                        "petri_dish": dish.name,
                        "image_id": image_id,
                        "image_path": str(image_lookup[image_id]),
                        "mask_path": str(ann_file),
                        "root_type": root_type,
                        "plant_number": int(plant_num)
                    })
    df = pd.DataFrame(records)
    return df


def detect_extract(im_path):
    """
    Find the important area of an image and extract it.

    Parameters:
        im_path (str): Path to the input image.
    
    Returns:
        np.ndarray: Extracted region of interest from the input image.
    """
    im = load_image(im_path)
    original_shape = im.shape

    _, im_thresh = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Find contours for binary threshold image.
    contours, _ = cv2.findContours(im_thresh, cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    # Get the largest contour, this should identify the petri dish.
    largest_contour = max(contours, key=cv2.contourArea)
    # Get bounding box coordinates for the largest contour.
    x, y, w, h = cv2.boundingRect(largest_contour)
    # Get the longest side length.
    l = max(w, h)
    # Ensure lengths are the same.
    c_x = x + w // 2
    c_y = y + h // 2
    # Get new coordinates.
    x = max(c_x - l // 2, 0)
    y = max(c_y - l // 2, 0)
    # Ensure coordinates do not exceed image dimensions.
    x = min(x, im.shape[1] - l)
    y = min(y, im.shape[0] - l)

    return im[y:y+l, x:x+l], x, y, l, original_shape


def padder(image: np.ndarray, patch_size: int) -> np.ndarray:
    """
    Pad the input image to ensure its dimensions are divisible by the specified patch size.
    
    Parameters:
        image (np.ndarray): Input image to be padded.
        patch_size (int): Size of the patches for which the image should be divisible.
    
    Returns:
        np.ndarray: Padded image with dimensions divisible by the patch size.
    """
    h, w = image.shape[:2]

    hp = ((h // patch_size) + 1) * patch_size - h
    wp = ((w // patch_size) + 1) * patch_size - w

    top_p = int(hp / 2)
    # bot_p = wp - top_p
    bot_p = hp - top_p

    lp = int(wp / 2)
    rp = wp - lp
    pad_im = cv2.copyMakeBorder(image, top_p, bot_p,
                                      lp, rp, cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE)
    return pad_im


def patch_image(image: np.ndarray, patch_size: int) -> np.ndarray:
    """
    Patchify the input image into smaller patches of the specified size.

    Parameters:
        image (np.ndarray): Input image to be patchified.
        patch_size (int): Size of the patches to create.
    
    Returns:
        np.ndarray: Array of image patches and the original shape of the image.
    """
    patches = patchify(image, (patch_size, patch_size), step=patch_size)
    patch_shape = patches.shape
    patches = patches.reshape(-1, patch_size, patch_size)
    return patches, patch_shape


def remove_noise(
    mask: np.ndarray, min_area: int = 50,
    apply_closing: bool = True,
    closing_iterations: int = 2
) -> np.ndarray:
    """
    Combined noise removal with opening, component filtering, and closing.
    
    This approach:
    1. Opens the mask to remove small isolated noise
    2. Filters components by minimum area
    3. Closes the mask to fill small holes
    
    Parameters:
        mask (np.ndarray)
            Binary mask to be denoised.
        min_area (int, optional)
            Minimum area for connected components to be retained. Defaults to 50.
        apply_closing (bool, optional)
            Whether to apply morphological closing after component filtering. Defaults to True.
        closing_iterations (int, optional)
            Number of iterations for morphological closing. Defaults to 2.
        
    Returns:
        np.ndarray: Denoised binary mask.
    """
    # Ensure binary
    if mask.max() > 1:
        mask = (mask > 0).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)
    
    # Morphological opening
    kernel_open = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    
    # Connected component filtering
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    
    denoised_mask = np.zeros_like(mask)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            denoised_mask[labels == label] = 1
    
    # Morphological closing to fill small holes
    if apply_closing and closing_iterations > 0:
        kernel_close = np.ones((3, 3), np.uint8)
        denoised_mask = cv2.morphologyEx(
            denoised_mask,
            cv2.MORPH_CLOSE, 
            kernel_close, iterations=closing_iterations
        )
    
    return denoised_mask