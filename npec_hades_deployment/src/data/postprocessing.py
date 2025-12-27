import os

import cv2
import numpy as np

from logger_config import logger

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
    
    Authors: Jason van Hamond, 232567@buas.nl
    
    :param mask: Binary mask as numpy array
    :param min_area: Minimum area for connected components
    :param apply_closing: Whether to apply morphological closing at the end
    :param closing_iterations: Number of closing iterations
    :return: Denoised binary mask
    """
    logger.info("Applying combined noise removal...")
    
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
    
    logger.info("Combined noise removal complete")
    return denoised_mask
