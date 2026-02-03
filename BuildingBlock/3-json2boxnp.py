import json
import numpy as np
import os

boxJson_floder = "./BoxCenterSizeLabelNorm"
boxNp_floder = "./BoxCenterSizeLabelNp"
data_half = False

material_class = {
    "accessoryMaterial": 0,
    "awningMaterial": 1,
    "balconyMaterial": 2,
    "chimneyMaterial": 3,
    "doorMaterial": 4,
    "floorMaterial": 5,
    "pillarMaterial": 6,
    "pipeMaterial": 7,
    "railingMaterial": 8,
    "roofMaterial": 9,
    "stairMaterial": 10,
    "wallMaterial": 11,
    "wallWithManyWindowMaterial": 11,
    "windowMaterial": 12,
}

import os

# specify the folder containing the JSON files

# get list of all files in the folder
files = os.listdir(boxJson_floder)

# filter out files that are not .json and remove the .json extension
filenames = [f[:-5] for f in files if f.endswith(".json")]

# write filenames to a text file
with open("building_train.lst", "w") as file:
    for name in filenames:
        file.write(name + "\n")


def to_one_hot(number, num_classes=13 + 2):
    one_hot = np.zeros(num_classes)
    one_hot[number] = 1
    return one_hot


os.makedirs(boxNp_floder, exist_ok=True)
json_file = [file for file in os.listdir(boxJson_floder) if file.endswith(".json")]


box_np = []

for file in json_file:
    box_class_list = []
    box_size_list = []
    box_location_list = []
    with open(os.path.join(boxJson_floder, file), "r") as f:
        boxes = json.load(f)
        for box in boxes:
            box_class = np.expand_dims(
                to_one_hot(material_class[box["materials"][0]]), 0
            )

            box_size = np.expand_dims(np.array(box["actor_size"]), 0)

            box_location = np.expand_dims(np.array(box["actor_location"]), 0)

            box_class_list.append(box_class)
            box_size_list.append(box_size)
            box_location_list.append(box_location)

        box_class = np.concatenate(box_class_list, 0)
        box_size = (
            np.concatenate(box_size_list, 0) / 2
            if data_half
            else np.concatenate(box_size_list, 0)
        )
        box_location = np.concatenate(box_location_list, 0)
        box_angles = np.zeros([box_class.shape[0], 1])
        path_building = os.path.join(boxNp_floder, file.replace(".json", ""))
        os.makedirs(path_building, exist_ok=True)
        np.savez(
            os.path.join(path_building, "boxes.npz"),
            class_labels=box_class,
            translations=box_location,
            sizes=box_size,
            angles=box_angles,
        )
