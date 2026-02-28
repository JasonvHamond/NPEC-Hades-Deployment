import numpy as np
import pandas as pd
from datetime import datetime
import xml.dom.minidom
from skan import Skeleton

ROOT_LABEL = 'A'
PLANT_LABEL = 'P'


def prepare_data_for_rxml(subset_plant: pd.DataFrame, skeleton_ob_loc: Skeleton) -> pd.DataFrame:
    rows = []

    skeleton_ids = []

    for plant in subset_plant['plant'].unique():
        plant_data = subset_plant.loc[subset_plant['plant'] == plant]
        primary_plant_data = plant_data.loc[plant_data['root_id'] == 'Primary']
        skeleton_id = primary_plant_data["skeleton-id"].values
        if skeleton_id.shape[0] > 0:
            skeleton_ids.append(skeleton_id[0])

    for index, row in subset_plant.iterrows():
        if row["skeleton-id"] not in skeleton_ids:
            continue
        yx = skeleton_ob_loc.path_coordinates(index)
        for y, x in yx:
            new_row = {"plant": row["plant"], "root_id": row["root_id"], "x": x, "y": y, "z": 0}
            rows.append(new_row)

    rxml_data = pd.DataFrame(rows, columns=["plant", "root_id", "x", "y", "z"])
    rxml_data['plant_id'] = rxml_data['plant']
    rxml_data['type'] = rxml_data['root_id']

    rxml_data.drop(columns=['plant', 'root_id'], inplace=True)
    return rxml_data


def create_plant(plant_id: int, all_coordinates: pd.DataFrame) -> str:
    """
    This function creates a plant entity with the provided plant ID and coordinates.

    :param plant_id: The ID of the plant to create.
    :param all_coordinates: DataFrame containing all coordinates related to the plant.
    :return: A string indicating the status of the plant creation process.
    """
    if 'x' not in all_coordinates.columns \
            or 'y' not in all_coordinates.columns \
            or 'z' not in all_coordinates.columns \
            or 'type' not in all_coordinates.columns:
        return None
    primary_root_coordinates = all_coordinates.loc[all_coordinates['type'] == 'Primary']
    plant_structure = f'''<plant id="{plant_id}" label="{PLANT_LABEL}"><root id="{plant_id}_1" label="{ROOT_LABEL}">'''
    plant_structure += generate_geometry(x=primary_root_coordinates['x'].values,
                                         y=primary_root_coordinates['y'].values,
                                         z=primary_root_coordinates['z'].values)

    unique_lateral_roots = all_coordinates.loc[all_coordinates['type'].str.contains('Lateral', na=False), 'type'].unique()

    for idx, el in enumerate(unique_lateral_roots):
        lateral_root_coordinates = all_coordinates.loc[all_coordinates['type'] == el]
        plant_structure += generate_lroot(id=idx + 1, label=ROOT_LABEL, x=lateral_root_coordinates['x'].values,
                                          y=lateral_root_coordinates['y'].values,
                                          z=lateral_root_coordinates['z'].values)

    plant_structure += '''</root></plant>'''

    return plant_structure


def generate_lroot(id: int, label: str, x: np.array, y: np.array, z: np.array) -> str:
    """
    This function generates a labeled root structure using the provided ID, label, and coordinates.

    :param id: The ID of the root structure.
    :param label: The label or name of the root structure.
    :param x: The array of x-coordinates of the root structure.
    :param y: The array of y-coordinates of the root structure.
    :param z: The array of z-coordinates of the root structure.
    :return: A string representing the root structure.
    """
    root_content = f'''<root id="{id}" label="{label}">'''
    root_content += generate_geometry(x, y, z)
    root_content += '</root>'
    return root_content


def generate_geometry(x: np.array, y: np.array, z: np.array) -> str:
    """
    This function generates a geometry using the provided arrays of x, y, and z coordinates.

    :param x: The array of x-coordinates.
    :param y: The array of y-coordinates.
    :param z: The array of z-coordinates.
    :return: A string representing the geometry structure.
    """
    if x.shape != y.shape and x.ndim != y.ndim != 1 and x.shape[0] != 0:
        raise ValueError("Invalid input arguments")

    array_size = x.shape[0]

    geometry_content = '''<geometry><polyline>'''
    for idx in range(0, array_size):
        geometry_content += f'{generate_point(x=x[idx], y=y[idx], z=z[idx])}'
    geometry_content += '''</polyline></geometry>'''

    return geometry_content


def generate_point(x: int, y: int, z: int) -> str:
    """
    This function generates a point with the provided x, y, and z coordinates.

    :param x: The x-coordinate of the point.
    :param y: The y-coordinate of the point.
    :param z: The z-coordinate of the point.
    :return: A string representing the format of the point.
    """
    if x < 0 or y < 0:
        raise ValueError("Invalid input arguments")
    return f'<point x="{x}" y="{y}" z="{z}"/>'


def generate_rsml(petri_dish_data: pd.DataFrame) -> str:
    """
    This function generates an RSML file based on the provided petri dish data.

    :param petri_dish_data: DataFrame containing data related to the petri dish experiment.
    :return: A string representing the RSML file .
    """
    scene_content = ''

    unique_plants = petri_dish_data['plant_id'].unique()
    for idx, el in enumerate(unique_plants):
        plant_data = petri_dish_data.loc[petri_dish_data['plant_id'] == el]
        scene_content += create_plant(idx, plant_data)

    return f'''<?xml version="1.0" encoding="UTF-8"?>
<rsml xmlns:po="http://www.plantontology.org/xml-dtd/po.dtd">
<metadata>
<version>1</version>
<unit>pixel</unit>
<resolution>1</resolution>
<last-modified>2014-02-26T00:00:00</last-modified>
<software>smartroot</software>
<user>mikepound</user>
<file-key>RSML-example:arabidopsis_simple</file-key>
<image>
<name>arabidopsis</name>
<captured>{datetime.now()}</captured>
</image>
</metadata>
<scene>
{scene_content}
</scene>
</rsml>'''.replace('\n', '').lstrip()


def write_rsml(rsml: str, save_path: str) -> None:
    """
    This function writes the RSML data to a file.

    :param rsml: The RSML data to write.
    :return: None
    """
    dom = xml.dom.minidom.parseString(rsml.strip())
    pretty_string = dom.toprettyxml().encode()

    with open(save_path, 'wb') as f:
        f.write(bytes(pretty_string))


def get_rsml(subset_branch: pd.DataFrame, skelton_ob_loc: Skeleton, save_path: str) -> None:
    """
    Main function for generating RSML data.

    :param subset_branch: DataFrame containing the subset of the branch data.
    :param skelton_ob_loc: Skeleton object containing the skeleton data.
    :param save_path: The path to save the RSML file.
    :return: None

    """
    rxml_data = prepare_data_for_rxml(subset_branch, skelton_ob_loc)
    rsml = generate_rsml(rxml_data)
    write_rsml(rsml, save_path)
