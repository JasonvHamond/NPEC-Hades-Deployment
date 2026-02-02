import os
import shutil
import threading
import time
import cv2
import numpy as np
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from utils.logger_config import logger
from PIL import Image
import pillow_jxl


def create_folder(folder_name: str) -> None:
    """
    Create a folder if it doesn't exist.

    Author: Vlad Matache 224108@buas.nl
    :param folder_name: Name of the folder to create.
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        logger.info(f"Folder '{folder_name}' created successfully.")
    else:
        logger.info(f"Folder '{folder_name}' already exists.")


def check_input_type(input_data: any) -> str:
    """
    Check the type of the input data.

    Author: Vlad Matache, 224108@buas.nl
    :param input_data: Input data to check.
    """
    if type(input_data) is str:
        return "str"
    elif type(input_data) is np.ndarray:
        return "np.ndarray"
    else:
        return "unsupported"


def load_image(image_path: str, verbose: bool = True) -> np.ndarray:
    """
    Load an image from a given path.

    Author: Vlad Matache, 224108@buas.nl
    :param image_path: Path of the image to load.
    :param verbose: Print additional information.
    :return: Loaded image.
    """
    if check_input_type(image_path) != "str":
        error_message = f"Unsupported input type: {check_input_type(image_path)}"
        logger.error(error_message)
        raise ValueError(error_message)
    else:
        image = cv2.imread(image_path)
        if image is None:
            error_message = f"Image not found at: {image_path}"
            logger.error(error_message)
            raise ValueError(error_message)
        elif len(image.shape) != 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        if verbose:
            logger.info(f"Loaded image with shape: {image.shape}")
        return image


def load_jxl_image(image_path: str) -> np.ndarray:
    """
    Load a JXL image from a given path.

    Author: Jason van Hamond

    Parameters:
    image_path (str): Path of the JXL image to load.

    Returns:
    np.ndarray: Loaded image.
    """
    # Check if the input type is correct.
    if check_input_type(image_path) != "str":
        logger.error(f"Unsupported input type: {check_input_type(image_path)}")
        raise ValueError(f"Unsupported input type: {check_input_type(image_path)}")
    else:
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


def load_images_from_folder(folder_path: str, verbose: bool = True) -> tuple[list, list]:
    """
    Load images from a given folder.

    Author: Vlad Matache, 224108@buas.nl
    :param folder_path: Path of the folder containing the images.
    :param verbose: Print additional information.
    :return: List of loaded images.
    """
    if check_input_type(folder_path) != "str":
        error_message = f"Unsupported input type: {check_input_type(folder_path)}"
        logger.error(error_message)
        raise ValueError(error_message)
    else:
        images = []
        filenames = []
        paths = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                image = load_image(os.path.join(folder_path, filename), verbose)
                images.append(image)
                filenames.append(filename)
                paths.append(os.path.join(folder_path, filename))
            elif filename.endswith(".jxl"):
                image = load_jxl_image(os.path.join(folder_path, filename))
                images.append(image)
                filenames.append(filename)
                paths.append(os.path.join(folder_path, filename))
        if verbose:
            logger.info(
                f"Loaded {len(images)} images from folder: {folder_path}"
            )
        return images, filenames, paths


def structure_folders(folder_path: str, name_convention: bool) -> None:
    """
    This function groups the petri dish images together in folders.
    It checks the ID and based on that groups them together in folders.

    Author: Wesley van Gaalen, 224682@buas.nl

    :param folder_path: The path to the folder containing the images.
    :return: None
    """

    if not name_convention:
         
        for x in os.listdir(folder_path):
             if x != '.DS_Store':

                petri_id = x.split(".")[0]
                 
                destination_folder = f"{folder_path}/../timeseries"
                destination = f"{destination_folder}/{petri_id}"
                source = f"{folder_path}/{x}"
                if not os.path.exists(destination):
                    os.makedirs(destination)
                shutil.copy(source, destination)
        return destination_folder
    else:



        for x in os.listdir(folder_path):
            if x!='.DS_Store':
                new_x = x.replace("_", "-")
                petri_id = new_x.split("-", 3)[3]
                petri_id = petri_id.rsplit('.', 1)[0]

                if not petri_id.startswith("ROOT"):
                    petri_id = "ROOT1-" + petri_id

                saving_id = x.split('-')[0] + '-' + x.split('-')[1]

                source = f"{folder_path}/{x}"
                destination_folder = f"{folder_path}/../timeseries"
                destination = f"{destination_folder}/{petri_id}"

                timeserie = new_x.split("-")[2]

                if not timeserie.isdigit():
                    timeserie = new_x.split("-")[1]

                if not os.path.exists(destination):
                    os.makedirs(destination)

                # Construct the new filename with .png extension
                new_filename = f"{saving_id}.png"
                destination_file = f"{destination}/{new_filename}"

                # Copy the file with the new name
                shutil.copy(source, destination_file)

        return destination_folder


def run_with_progress_bar(func, *args, **kwargs):
    def target():
        func(*args, **kwargs)

    thread = threading.Thread(target=target)
    thread.start()

    with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            transient=True,
    ) as progress:
        task = progress.add_task("Processing...", total=100)

        while thread.is_alive():
            time.sleep(0.1)
            progress.update(task, advance=1)
            if progress.tasks[task].completed >= 100:
                progress.update(task, completed=0)

        thread.join()
        progress.update(task, completed=100)
