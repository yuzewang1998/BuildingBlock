import json
import numpy as np
import os

boxJson_floder = "./BoxCenterSizeLabel_all"
boxJsonAugment_floder = "./BoxCenterSizeLabelAugment"

os.makedirs(boxJsonAugment_floder, exist_ok=True)
json_file = [file for file in os.listdir(boxJson_floder) if file.endswith(".json")]
# print(json_file)
box_np = []


def rotation_matrix_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


angles = [90, 180, 270, 360]

rotates = [90, 90, 90, 90]

radians = [np.radians(rotate) for rotate in rotates]

mirror = [False, True]

for file in json_file:
    box_class_list = []
    box_size_list = []
    box_location_list = []
    with open(os.path.join(boxJson_floder, file), "r") as f:
        boxes = json.load(f)
    for whether_mirror in mirror:
        for angle, theta in zip(angles, radians):
            rot_mat = rotation_matrix_z(theta)

            for actor in boxes:
                if whether_mirror and angle == 90:
                    actor["actor_location"][0] = -actor["actor_location"][0]
                location = np.array(actor["actor_location"])
                new_location = rot_mat.dot(location)
                actor["actor_location"] = new_location.tolist()

                size = np.array(actor["actor_size"])
                if angle != 0:
                    actor["actor_size"][:2] = [size[1], size[0]]

            with open(
                os.path.join(
                    boxJsonAugment_floder,
                    file.replace(".json", f"_A{angle}_mirror{whether_mirror}.json"),
                ),
                "w",
            ) as json_file:
                json.dump(boxes, json_file, indent=4)
