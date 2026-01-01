from opentrons import protocol_api
from opentrons.types import Location, Point
import opentrons.execute
import time
import csv

metadata = {
    "apiLevel": "2.13",
    "protocolName": "Testing Robotic Environment Limits",
    "description": """This protocol moves the pipette to a specific target_location.""",
    "author": "Vlad Matache",
    "improvements": "Jason van Hamond"
}

def define_protocol():
    protocol = opentrons.execute.get_protocol_api("2.13")

    # Set motor speeds.
    protocol.max_speeds["A"] = 100
    protocol.max_speeds["X"] = 600
    protocol.max_speeds["Y"] = 400
    protocol.max_speeds["Z"] = 100

    return protocol


def load_labware(protocol):
    # Load labware and pipette
    tips = protocol.load_labware("opentrons_96_tiprack_10ul", 1)
    plate = protocol.load_labware("corning_96_wellplate_360ul_flat", 7)
    pipette = protocol.load_instrument("p20_single_gen2", "right", tip_racks=[tips])

    middle_slot = [330, 131, 70]
    origin_slot = [265, 195, 75]
    seed_loc = [329.5, 131.8125, 75]

    return tips, plate, pipette, middle_slot, origin_slot, seed_loc


def get_innoculation_point(x,y, origin_slot):
    # Coordinates for custom innouclation point.
    landmark_x = x - 565
    x_conversion = 24
    landmark_y = y - 41
    y_conversion = 24

    # Converting to robot coordinate system (mm), last value in the list is the height
    innoculation_point = [
        landmark_x / x_conversion + origin_slot[0] + 6.8,
        origin_slot[1] - landmark_y / y_conversion - 5,
        64.6
    ]
    return innoculation_point


def export_csv(csv_path, data):
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = [
            "iteration",
            "coords",
            "start_time",
            "end_time",
            "duration_s"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

    print(f"Performance data saved to {csv_path}.")


def run(locations, filename=None):
    """
    """
    protocol = define_protocol()
    tips, plate, pipette, _, origin_slot, _ = load_labware(protocol)
    if filename == None:
        filename = "run1"
    # Log the start time of full protocol.
    runtimes = {"start_all": time.time()}
    # Home the pipette.
    protocol.home()
    performance_data = []
    for i, location in enumerate(locations, start=1):
        # Calculate the inoculation point.
        coords = get_innoculation_point(location[0], location[1], origin_slot)
        # Start the iteration's timer
        iter_start = time.time()
        # Select current tip based on it index.
        curr_tip = f"A{i}"
        curr_well = f"A{i+1}"
        # Pick up a tip
        pipette.pick_up_tip(tips[curr_tip]) #Slot on tip rack A-Z: row 1-12: Column
        source_well = plate[curr_well] #Slot in well rack A-Z: row 1-12: Column
        pipette.aspirate(10, source_well, rate=1.2)
        # Set coordinates for target_location
        x = coords[0]
        y = coords[1]
        
        # Movement of robot to target Location.
        z = 150
        target_location = Location(Point(x, y, z), None)
        pipette.move_to(target_location)
        z = location[2]
        target_location = Location(Point(x, y, z), None)
        pipette.move_to(target_location)
        pipette.dispense(20, target_location, rate=1.5)
        time.sleep(1)
        target_location = Location(Point(x, y, z), None)
        pipette.move_to(target_location)
        gantry_position = protocol._core.get_hardware().gantry_position(
            pipette._core.get_mount()
        )
        print(str(gantry_position))
        
        # Dispose of the tip into the trash chute.
        pipette.drop_tip(home_after=False)
        # Calculate the iteration's full time.
        iter_end = time.time()
        duration = iter_end - iter_start
        # Append results to performance data list.
        performance_data.append({
            'iteration': i,
            'coords': coords,
            'start_time': iter_start,
            'end_time': iter_end,
            'duration_s': duration
        })
        # Also record runtimes dictionary for legacy calculations
        runtimes[f"coords_{i}_start"] = iter_start
        runtimes[f"coords_{i}_end"] = iter_end
    # Home robot to calibrate after sequence.
    protocol.home()
    # Track end time of full protocol.
    runtimes["end_all"] = time.time()

    # Save all times to CSV
    export_csv(f"Runtimes/{filename}.csv", runtimes)
