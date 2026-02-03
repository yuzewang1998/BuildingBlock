import json
import numpy as np
import os
import matplotlib.pyplot as plt

boxJson_floder = "./BoxCenterSizeLabelAugment"
boxNormJson_floder = "./BoxCenterSizeLabelNorm"
os.makedirs(boxNormJson_floder, exist_ok=True)

material_class = {
    "accessoryMaterial": 0,
    "awaingMaterial": 1,
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
    "windowMaterial": 12,
}

json_file = [file for file in os.listdir(boxJson_floder) if file.endswith(".json")]
json_file = sorted(json_file)
box_np = []


for file in json_file:
    box_class_list = []
    box_size_list = []
    box_location_list = []
    with open(os.path.join(boxJson_floder, file), "r") as f:

        boxes = json.load(f)

        min_coords = [float("inf")] * 3
        max_coords = [float("-inf")] * 3

        for item in boxes:
            location = item["actor_location"]
            size = item["actor_size"]

            min_vertex = [location[i] - size[i] / 2 for i in range(3)]
            max_vertex = [location[i] + size[i] / 2 for i in range(3)]

            for i in range(3):
                min_coords[i] = min(min_coords[i], min_vertex[i])
                max_coords[i] = max(max_coords[i], max_vertex[i])

        range_max = max(max_coords[i] - min_coords[i] for i in range(3))
        for item in boxes:
            location = item["actor_location"]
            size = item["actor_size"]
            normalized_location = [
                (location[i] - min_coords[i]) / range_max
                + 0.5 * (range_max - (max_coords[i] - min_coords[i])) / range_max
                for i in range(3)
            ]

            normalized_size = [s / range_max for s in size]
            item["actor_location"] = normalized_location
            item["actor_size"] = normalized_size

        with open(os.path.join(boxNormJson_floder, file), "w") as f:
            json.dump(boxes, f, indent=4)
