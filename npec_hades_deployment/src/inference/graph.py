import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np


def get_timeserie_num(string: str) -> str:
    """
    This function takes the file name of petri dishes in and finds the timeserie number in it.
    The petri dishes have 2 types of locations for the timeserie number. This function finds the correct one, no matter which one it is.
    :param string: File name of the petri dish.
    :return: timeserie: The timeserie number of the petri dish.
    :example:
    >>> get_timeserie_num("034_43-13-ROOT1-2023-08-08_control_pH7_-Fe+B_col0_02-Fish Eye Corrected.png")
    13
    >>> get_timeserie_num("43-11-ROOT1-2023-08-08_pvdCherry_OD01_f6h1_03-Fish Eye Corrected.png")
    11
    """
    # Replace the _ with - to make it easier to split the string
    new_x = string.replace("_", "-")
    # Split the string by the - character
    timeserie = new_x.split("-")[1]

    # If the timeserie is not a digit, it means that the timeserie is in another location in the string.
    if not timeserie.isdigit():
        # Split the string by the - character
        timeserie = new_x.split("-")[1]

    return timeserie


def add_petri_dish(petri_dishes: dict, num: int) -> dict:
    """
    This function adds a new petri dish to the dictionary of petri dishes.
    :param petri_dishes: Dictionary of petri dishes.
    :param num: Number of the petri dish.
    :return: petri_dishes: Dictionary of petri dishes.
    :example:
    >>> add_petri_dish({}, 1)
    {1: {}}
    """
    # Convert the number to an integer
    key = int(num)
    # Add the petri dish to the dictionary
    petri_dishes[key] = {}
    return petri_dishes


def add_plant_petri_dish(petri_dishes: dict, num: int, plant: int, primary_length: float, lateral_length: float,
                         total_length: float) -> dict:
    """
    This function adds a plant to the petri dish in the dictionary of petri dishes.
    It also adds the primary, lateral, and total root length of the plant to the petri dish.
    :param petri_dishes: Dictionary of petri dishes.
    :param num: Number of the petri dish.
    :param plant: Number of the plant.
    :param primary_length: Primary root length of the plant.
    :param lateral_length: Lateral root length of the plant.
    :param total_length: Total root length of the plant.
    :return: petri_dishes: Dictionary of petri dishes.
    :example:
    >>> add_plant_petri_dish({1: {}}, 1, 1, 100, 50, 150)
    {1: {1: {'primary_length': 100, 'lateral_length': 50, 'total_length': 150}}}
    """
    key = int(num)
    petri_dishes[key][plant] = {
        "primary_length": primary_length,
        "lateral_length": lateral_length,
        "total_length": total_length
    }
    return petri_dishes


def find_num_of_plants(dictionary: dict) -> int:
    # Find the amount of plants in the petri dishes.
    num_of_plants = 0
    # Loop over each petri dish
    for key in dictionary:
        # Check if the petri dish has more plants than the last one
        if len(dictionary[key]) > num_of_plants:
            num_of_plants = len(dictionary[key])
    return num_of_plants


def plot_timeline_graph(dictionary: dict, folder_dir: str):
    """
    This function plots the root length of each plant in the petri dishes over time.
    It saves the plots in the assets folder of the timeline folder.
    It measures the primary, lateral, and total root length per plant.
    :param dictionary: Dictionary of the petri dishes.
    :param folder_dir: Directory of the timeline folder.
    """
    # Find the amount of plants in the petri dishes
    find_num_of_plants(dictionary)
    keys = []

    for key in dictionary:
        for k in dictionary[key]:
            if k not in keys:
                keys.append(int(k))
    # Loop over all the plants
    for plant in keys:
        # Create the lists for the primary, lateral, and total root length, and the timelines
        primary_length = []
        lateral_length = []
        total_length = []
        timelines = []
        # Loop over all the petri dishes, sorted by the timeline/day.
        for timeline in sorted(dictionary.keys()):
            # If the petri dish doesn't have as many plants as the total number of plants, add empty plants to the petri dish.
            plants = dictionary[timeline].keys()
            for x in keys:
                if x not in plants:
                    add_plant_petri_dish(dictionary, timeline, x, 0, 0, 0)

            # Append the primary, lateral, and total root length of the plant to the lists.
            # print(dictionary)
            # print(f"timeline: {timeline}, plant: {plant}")
            primary_length.append(
                dictionary[timeline][plant]["primary_length"])
            lateral_length.append(
                dictionary[timeline][plant]["lateral_length"])
            total_length.append(dictionary[timeline][plant]["total_length"])
            timelines.append(int(timeline))
        # Create the plot
        plt.plot(timelines, primary_length,
                 label="Primary root length", color="red", marker='o')
        plt.plot(timelines, lateral_length,
                 label="Lateral root length", color="orange", marker='o')
        plt.plot(timelines, total_length, label="Total root length",
                 color="black", marker='o')
        plt.title(f"Root length for plant: {plant}")
        plt.xlabel("Days")
        plt.ylabel("Length (mm)")
        plt.legend()
        plt.yticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.ylim(bottom=0, top=1000)
        plt.xticks([0, 5, 10, 15, 20])
        plt.xlim(left=0, right=20)
        plt.legend(loc='upper left')
        plt.savefig(f"{folder_dir}/plant_{plant}.png")
        plt.clf()


def plot_primary_graph(dictionary: dict, folder_dir: str):
    """
    This function plots the primary root length of each plant in the petri dishes over time.
    It saves the plots in the assets folder of the timeline folder.
    It measures the primary root length per plant.
    :param dictionary: Dictionary of the petri dishes.
    :param folder_dir: Directory of the timeline folder.
    """

    # Find the amount of plants in the petri dishes
    num_of_plants = find_num_of_plants(dictionary)

    timelines = []
    primary_length = np.zeros((len(dictionary), num_of_plants))
    x = 0
    for timeline in sorted(dictionary.keys()):
        x += 1
        timelines.append(timeline)
        y = 0
        for plant in dictionary[timeline]:
            primary_length[x - 1][int(y)] = dictionary[timeline][plant]["primary_length"]
            y += 1
    # Create the plot1
    # print(primary_length[0])
    for plant in range(num_of_plants):
        plt.plot(timelines, primary_length[:, plant], marker='o', label=f"Plant {plant + 1}")
    plt.title("Primary root length for all plants")
    plt.xlabel("Days")
    plt.ylabel("Length (mm)")
    plt.legend(loc='upper left')
    plt.yticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(bottom=0, top=1000)
    plt.xticks([0, 5, 10, 15, 20])
    plt.xlim(left=0, right=20)
    plt.savefig(f"{folder_dir}/primary_length.png")
    plt.clf()


def plot_lateral_graph(dictionary: dict, folder_dir: str):
    """
    This function plots the lateral root length of each plant in the petri dishes over time.
    It saves the plots in the assets folder of the timeline folder.
    It measures the lateral root length per plant.
    :param dictionary: Dictionary of the petri dishes.
    :param folder_dir: Directory of the timeline folder.
    """

    # Find the amount of plants in the petri dishes
    num_of_plants = find_num_of_plants(dictionary)

    timelines = []
    lateral_length = np.zeros((len(dictionary), num_of_plants))
    x = 0
    for timeline in sorted(dictionary.keys()):
        x += 1
        timelines.append(timeline)
        y = 0
        for plant in dictionary[timeline]:
            lateral_length[x - 1][int(y)] = dictionary[timeline][plant]["lateral_length"]
            y += 1
    # Create the plot1
    # print(primary_length[0])
    for plant in range(num_of_plants):
        plt.plot(timelines, lateral_length[:, plant], marker='o', label=f"Plant {plant + 1}")
    plt.title("Lateral root length for all plants")
    plt.xlabel("Days")
    plt.ylabel("Length (mm)")
    plt.legend(loc='upper left')
    plt.yticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(bottom=0, top=1000)
    plt.xticks([0, 5, 10, 15, 20])
    plt.xlim(left=0, right=20)
    plt.savefig(f"{folder_dir}/lateral_length.png")
    plt.clf()


def plot_total_graph(dictionary: dict, folder_dir: str):
    """
    This function plots the total root length of each plant in the petri dishes over time.
    It saves the plots in the assets folder of the timeline folder.
    It measures the total root length per plant.
    :param dictionary: Dictionary of the petri dishes.
    :param folder_dir: Directory of the timeline folder.
    """

    # Find the amount of plants in the petri dishes
    num_of_plants = find_num_of_plants(dictionary)

    timelines = []
    total_length = np.zeros((len(dictionary), num_of_plants))
    x = 0
    for timeline in sorted(dictionary.keys()):
        x += 1
        timelines.append(timeline)
        y = 0
        for plant in dictionary[timeline]:
            total_length[x - 1][int(y)] = dictionary[timeline][plant]["total_length"]
            y += 1
    # Create the plot1
    for plant in range(num_of_plants):
        plt.plot(timelines, total_length[:, plant], marker='o', label=f"Plant {plant + 1}")
    plt.title("Total root length for all plants")
    plt.xlabel("Days")
    plt.ylabel("Length (mm)")
    plt.legend(loc='upper left')
    plt.yticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(bottom=0, top=1000)
    plt.xticks([0, 5, 10, 15, 20])
    plt.xlim(left=0, right=20)
    plt.savefig(f"{folder_dir}/total_length.png")
    plt.clf()


def create_timeline_graph(folder_dir: str):
    """
    This function is the main function for creating the timeline graph.
    It loops over all the timeline folders and creates the timeline graph for each petri dish. One graph per plant.

    :param folder_dir: Directory of the timeline folders.
    :example:
    >>> are_consecutive(timeline)
    """
    # Loop over all the timeline folders
    for timeline in os.listdir(folder_dir):
        # Create the dictionary of the petri dishes
        petri_dish_dict = {}
        # Loop over all the petri dishes in the timeline folder
        for petri_dish in os.listdir(f"{folder_dir}/{timeline}"):
            # If the petri dish is a folder, and not the assets folder
            if os.path.isdir(f"{folder_dir}/{timeline}/{petri_dish}") and petri_dish != "assets":
                # Get the timeserie number/ the day of the petri dish
                timeserie = get_timeserie_num(petri_dish)
                # Add the day to the dictionary
                petri_dish_dict = add_petri_dish(petri_dish_dict, timeserie)
                # Read the excel file with the root length measurements
                df = pd.read_excel(
                    f"{folder_dir}/{timeline}/{petri_dish}/measurements.xlsx")
                # Sort the dataframe by the plant number
                sorted_df = df.sort_values(by=["plant"])
                # Loop over all the plants in the petri dish
                for index, row in sorted_df.iterrows():
                    # If the plant is NaN, set it to 0
                    if pd.isna(row["plant"]):
                        row["plant"] = 0
                    # Add the plant to the dictionary
                    add_plant_petri_dish(
                        petri_dish_dict, timeserie, int(row["plant"]), row["Primary_length(mm)"],
                        row["Lateral_length(mm)"], row["Total_length(mm)"])
        # Create the assets folder
        os.makedirs(f"{folder_dir}/{timeline}/assets", exist_ok=True)
        # Plot the timeline graph
        plot_timeline_graph(petri_dish_dict, f"{folder_dir}/{timeline}/assets")
        plot_primary_graph(petri_dish_dict, f"{folder_dir}/{timeline}/assets")
        plot_lateral_graph(petri_dish_dict, f"{folder_dir}/{timeline}/assets")
        plot_total_graph(petri_dish_dict, f"{folder_dir}/{timeline}/assets")
