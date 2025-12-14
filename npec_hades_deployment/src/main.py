import os
import typer
import tensorflow as tf
from rich import print
from data.processing import padder
from inference.features import save_prediction_for_folder
from inference.graph import create_timeline_graph
from inference.roots_segmentation import measure_folder
from utils.helpers import create_folder
from utils.metrics import iou, f1

# Set the environment variable before importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = typer.Typer()


def get_roots_lengths(folder_dir, template_path, name_convention):
    print(" Segmenting roots and measuring lengths...")
    expected_centers = [
        (1000, 550, 1),
        (1500, 550, 2),
        (2000, 550, 3),
        (2500, 550, 4),
        (3000, 550, 5)
    ]
    _ = measure_folder(folder_dir, expected_centers)
    print(" Roots segmented and measured 'successfully'.")
    if name_convention:
        _ = create_timeline_graph(folder_dir)
        print(" Graphs created 'successfully'.")
        print("----")


def check_input_folder(populated):
    working_directory = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(working_directory, 'input')
        

    if populated and len(os.listdir(input_folder)) != 0:
        return True
    elif len(os.listdir('input')) == 0:
        print(
            "[bold green] Action Required: [/bold green]The input folder is [bold red]empty[/bold red] please place the input images in the created input folder.")
        populated = typer.confirm(' Did you add all the image and want to continue?')
        return check_input_folder(populated)
    else:
        print("[bold green] Action Required: [/bold green]Please place the input images in the created input folder.")
        populated = typer.confirm(' Did you add all the image and want to continue?')
        return check_input_folder(populated)


@app.command()
def main():
    """
    This function is the entry point of the CLI application.

    :param input_path: Path to the input image.
    """

    # Get the working directory of the script
    working_directory = os.path.dirname(os.path.abspath(__file__))
    
    # Create input folder in the working directory
    input_folder = os.path.join(working_directory, 'input')
    os.makedirs(input_folder, exist_ok=True)

    folder_dir = "timeseries"
    template_path = "assets/seeding_template.tif"

    # Loading models (choice of pre-built models or custom models)
    print("----")
    print(
        "[bold red] Alert! [/bold red]A folder called 'input' has been created in the current directory.")
    print(
        "[bold green] Action Required: [/bold green]Please place the input images in this folder!"
    )
    populated = typer.confirm(' Did you add all the image and want to continue?')
    if check_input_folder(populated):
        print("----")
    print("[bold red] Alert! [/bold red] Let's choose the model for segmentation now:")
    pre_built_models = typer.confirm(' Do you want to use pre-built models?')
    if pre_built_models:
        print("[bold red] Alert! [/bold red]The [bold green]pre-trained[/bold green] models will be used!")
        # Load the models
        root_segmentation_model = tf.keras.models.load_model(
            'model_refrence/model_root_14.h5', custom_objects={'f1': f1})
        shoot_segmentation_model = tf.keras.models.load_model(
            'model_refrence/model_shoot_10.h5', custom_objects={'f1': f1})

    else:
        print("[bold red]Alert! [/bold red]Your [bold green]own[/bold green] models will be used!")
        root_model_path = typer.prompt(
            '[bold green] Action Required: [/bold green]Please enter the full path to the root segmentation model')
        root_segmentation_model = tf.keras.models.load_model(
            root_model_path, custom_objects={'f1': f1, 'iou': iou})

        shoot_model_path = typer.prompt(
            '[bold green] Action Required: [/bold green] Please enter the full path to the shoot segmentation model')
        shoot_segmentation_model = tf.keras.models.load_model(
            shoot_model_path, custom_objects={'f1': f1, 'iou': iou})

    # Creating Masks
    print("[bold red] Alert! [/bold red]Models loaded 'successfully'.")
    print("----")
    name_convention = typer.confirm('Do you want to use the default naming convention?')
    typer.echo(" Creating Masks for the input images...")
    save_prediction_for_folder(input_folder, root_segmentation_model,
                               shoot_segmentation_model, padder, name_convention, verbose=False)
    typer.echo(" Masks created 'successfully'.")
    print("----")
    # Measuring Root Lengths
    get_roots_lengths(folder_dir, template_path, name_convention)

    print("[green]Phenotyping Completed Successfully![/green]")


if __name__ == "__main__":
    print("[green]Starting Pyphenotyper[/green]")
    app()
