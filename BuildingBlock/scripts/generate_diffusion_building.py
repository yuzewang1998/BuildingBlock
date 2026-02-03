#
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
#

"""Script used for generating scenes using a previously trained model."""
import argparse
import logging
import os
import sys

import numpy as np
import torch

from training_utils import load_config
from utils import floor_plan_from_scene, export_scene, get_textured_objects_in_scene

from scene_synthesis.datasets import filter_function, get_dataset_raw_and_encoded
from scene_synthesis.datasets.threed_front import ThreedFront
from scene_synthesis.datasets.threed_future_dataset import ThreedFutureDataset
from scene_synthesis.networks import build_network
from scene_synthesis.utils import (
    get_textured_objects,
    get_textured_objects_based_on_objfeats,
)
from scene_synthesis.stats_logger import AverageAggregator

from simple_3dviz import Scene

# from simple_3dviz.window import show
from simple_3dviz.behaviours.keyboard import SnapshotOnKey, SortTriangles
from simple_3dviz.behaviours.misc import LightToCamera
from simple_3dviz.behaviours.movements import CameraTrajectory
from simple_3dviz.behaviours.trajectory import Circle
from simple_3dviz.behaviours.io import SaveFrames, SaveGif
from simple_3dviz.utils import render
import matplotlib.pyplot as plt
from pyrr import Matrix44
from utils import render as render_top2down
from utils import merge_meshes
import trimesh
import open3d as o3d
from utils import merge_meshes, computer_intersection, computer_symmetry
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import seaborn as sns
import json


def categorical_kl(p, q):
    return (p * (np.log(p + 1e-6) - np.log(q + 1e-6))).sum()


def rotate_points(points, angle, axis="z"):

    angle_rad = np.deg2rad(angle)

    if axis == "x":
        rotation_matrix = np.array(
            [
                [1, 0, 0],
                [0, np.cos(angle_rad), -np.sin(angle_rad)],
                [0, np.sin(angle_rad), np.cos(angle_rad)],
            ]
        )
    elif axis == "y":
        rotation_matrix = np.array(
            [
                [np.cos(angle_rad), 0, np.sin(angle_rad)],
                [0, 1, 0],
                [-np.sin(angle_rad), 0, np.cos(angle_rad)],
            ]
        )
    elif axis == "z":
        rotation_matrix = np.array(
            [
                [np.cos(angle_rad), -np.sin(angle_rad), 0],
                [np.sin(angle_rad), np.cos(angle_rad), 0],
                [0, 0, 1],
            ]
        )

    rotated_points = [np.dot(rotation_matrix, point) for point in points]

    return rotated_points


def draw_bounding_box_line(ax, bbox, color="r"):
    if color == "black":
        return

    xmin, ymin, zmin, xmax, ymax, zmax = bbox

    edges = [
        [[xmin, ymin, zmin], [xmax, ymin, zmin]],
        [[xmin, ymin, zmin], [xmin, ymax, zmin]],
        [[xmin, ymin, zmin], [xmin, ymin, zmax]],
        [[xmax, ymin, zmin], [xmax, ymax, zmin]],
        [[xmax, ymin, zmin], [xmax, ymin, zmax]],
        [[xmin, ymax, zmin], [xmax, ymax, zmin]],
        [[xmin, ymax, zmin], [xmin, ymax, zmax]],
        [[xmax, ymax, zmin], [xmax, ymax, zmax]],
        [[xmin, ymin, zmax], [xmax, ymin, zmax]],
        [[xmin, ymin, zmax], [xmin, ymax, zmax]],
        [[xmin, ymax, zmax], [xmax, ymax, zmax]],
        [[xmax, ymin, zmax], [xmax, ymax, zmax]],
    ]
    for edge in edges:
        ax.plot3D(*zip(*edge), color=color)


def draw_bounding_box_box(
    ax, bbox, color="r", rotation_angle=0, rotation_axis="z", changeYZ=False
):
    if color == "black":
        return
    xmin, ymin, zmin, xmax, ymax, zmax = bbox

    vertices = [
        [
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmin],
        ],
        [
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
            [xmin, ymax, zmax],
        ],
        [
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmax, ymin, zmax],
            [xmin, ymin, zmax],
        ],
        [
            [xmin, ymax, zmin],
            [xmax, ymax, zmin],
            [xmax, ymax, zmax],
            [xmin, ymax, zmax],
        ],
        [
            [xmin, ymin, zmin],
            [xmin, ymax, zmin],
            [xmin, ymax, zmax],
            [xmin, ymin, zmax],
        ],
        [
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmax, ymax, zmax],
            [xmax, ymin, zmax],
        ],
    ]

    rotated_vertices = []
    for face in vertices:
        rotated_face = rotate_points(face, rotation_angle, axis=rotation_axis)
        if changeYZ:
            rotated_face = rotate_points(rotated_face, 90, axis="x")
        rotated_vertices.append(rotated_face)

    for face in rotated_vertices:
        poly3d = Poly3DCollection(
            [face], color=color, linewidths=1, edgecolors="k", alpha=0.3
        )
        ax.add_collection3d(poly3d)


material_assets = {
    "accessory": (0.58, 0, 0.9),
    "awning": (0.63, 0.42, 0.9),
    "balcony": (0.39, 0.21, 0.9),
    "chimney": (0.0, 0.32, 0.9),
    "door": (0.9, 0.77, 0.0),
    "floor": (0.0, 0.43, 0.9),
    "pillar": (0.39, 0.9, 0.0),
    "pipe": (0.9, 0.4, 0.0),
    "railing": (0.9, 0.0, 0.29),
    "roof": (0.9, 0.0, 0.0),
    "stair": (0.9, 0.0, 0.7),
    "wall": (0.0, 0.15, 0.9),
    "window": (0.0, 0.9, 0.0),
}


def iou_3d(box1, box2, eps=1e-8, mode="iou"):
    x_min1, x_max1, y_min1, y_max1, z_min1, z_max1 = box1
    x_min2, x_max2, y_min2, y_max2, z_min2, z_max2 = box2

    x_min_inter = max(x_min1, x_min2)
    y_min_inter = max(y_min1, y_min2)
    z_min_inter = max(z_min1, z_min2)

    x_max_inter = min(x_max1, x_max2)
    y_max_inter = min(y_max1, y_max2)
    z_max_inter = min(z_max1, z_max2)

    inter_volume = (
        max(0, x_max_inter - x_min_inter)
        * max(0, y_max_inter - y_min_inter)
        * max(0, z_max_inter - z_min_inter)
    )

    volume1 = (x_max1 - x_min1) * (y_max1 - y_min1) * (z_max1 - z_min1)
    volume2 = (x_max2 - x_min2) * (y_max2 - y_min2) * (z_max2 - z_min2)

    iou = inter_volume / (volume1 + volume2 - inter_volume + eps)
    overlap_ratio = inter_volume / (min(volume1, volume2) + eps)
    if mode == "iou":
        return iou
    else:
        return overlap_ratio


def nms_3d(boxes, iou_threshold=0.5):
    keep_indices = []
    while boxes:

        current_box = boxes[0]
        keep_indices.append(current_box)

        boxes = [box for box in boxes[1:] if iou_3d(current_box, box) < iou_threshold]

    return keep_indices


def merge_boxes(boxes):

    x_min = min([box[0] for box in boxes])
    x_max = max([box[1] for box in boxes])
    y_min = min([box[2] for box in boxes])
    y_max = max([box[3] for box in boxes])
    z_min = min([box[4] for box in boxes])
    z_max = max([box[5] for box in boxes])
    return (x_min, x_max, y_min, y_max, z_min, z_max)


def nms_and_merge(boxes, iou_threshold=0.3):
    merged_boxes = []
    boxes_idx = list(range(boxes["class_labels"].shape[1]))

    res_idx = []
    while boxes_idx:
        new_boxes_idx = []
        now_idx = boxes_idx[0]
        res_idx.append(now_idx)
        x_min, x_max, y_min, y_max, z_min, z_max = (
            boxes["translations"][0][now_idx][0] - boxes["sizes"][0][now_idx][0] / 2,
            boxes["translations"][0][now_idx][0] + boxes["sizes"][0][now_idx][0] / 2,
            boxes["translations"][0][now_idx][1] - boxes["sizes"][0][now_idx][1] / 2,
            boxes["translations"][0][now_idx][1] + boxes["sizes"][0][now_idx][1] / 2,
            boxes["translations"][0][now_idx][2] - boxes["sizes"][0][now_idx][2] / 2,
            boxes["translations"][0][now_idx][2] + boxes["sizes"][0][now_idx][2] / 2,
        )

        class_labels = torch.argmax(boxes["class_labels"][0][now_idx])
        to_merge = [now_idx]

        merge_bbx = [(x_min, x_max, y_min, y_max, z_min, z_max)]
        # import ipdb;ipdb.set_trace()

        for idx in boxes_idx[1:]:
            if torch.argmax(boxes["class_labels"][0][idx]) != class_labels:
                continue
            x_min_tmp, x_max_tmp, y_min_tmp, y_max_tmp, z_min_tmp, z_max_tmp = (
                boxes["translations"][0][idx][0] - boxes["sizes"][0][idx][0] / 2,
                boxes["translations"][0][idx][0] + boxes["sizes"][0][idx][0] / 2,
                boxes["translations"][0][idx][1] - boxes["sizes"][0][idx][1] / 2,
                boxes["translations"][0][idx][1] + boxes["sizes"][0][idx][1] / 2,
                boxes["translations"][0][idx][2] - boxes["sizes"][0][idx][2] / 2,
                boxes["translations"][0][idx][2] + boxes["sizes"][0][idx][2] / 2,
            )
            if (
                iou_3d(
                    (x_min, x_max, y_min, y_max, z_min, z_max),
                    (x_min_tmp, x_max_tmp, y_min_tmp, y_max_tmp, z_min_tmp, z_max_tmp),
                    mode="overlap_ratio",
                )
                > iou_threshold
            ):
                to_merge.append(idx)
                merge_bbx.append(
                    ((x_min_tmp, x_max_tmp, y_min_tmp, y_max_tmp, z_min_tmp, z_max_tmp))
                )

        boxes_idx = [idx for idx in boxes_idx if idx not in to_merge]

        merged_box = merge_boxes(merge_bbx)

        boxes["translations"][0][now_idx] = torch.tensor(
            [
                (merged_box[0] + merged_box[1]) / 2,
                (merged_box[2] + merged_box[3]) / 2,
                (merged_box[4] + merged_box[5]) / 2,
            ]
        )
        boxes["sizes"][0][now_idx] = torch.tensor(
            [
                merged_box[1] - merged_box[0],
                merged_box[3] - merged_box[2],
                merged_box[5] - merged_box[4],
            ]
        )

    new_boxes = {
        "class_labels": torch.zeros(1, 0, boxes["class_labels"].shape[-1]),
        "translations": torch.zeros(1, 0, boxes["translations"].shape[-1]),
        "sizes": torch.zeros(1, 0, boxes["sizes"].shape[-1]),
    }

    for idx in res_idx:
        new_boxes["class_labels"] = torch.cat(
            [
                new_boxes["class_labels"],
                boxes["class_labels"][0][idx].unsqueeze(0).unsqueeze(0),
            ],
            1,
        )
        new_boxes["translations"] = torch.cat(
            [
                new_boxes["translations"],
                boxes["translations"][0][idx].unsqueeze(0).unsqueeze(0),
            ],
            1,
        )
        new_boxes["sizes"] = torch.cat(
            [new_boxes["sizes"], boxes["sizes"][0][idx].unsqueeze(0).unsqueeze(0)], 1
        )

    return new_boxes


def draw_scene(boxes, classes, save_path="../draw.png", nms=None, size_half=False):
    fig = plt.figure()
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    if nms:
        boxes = nms_and_merge(boxes)
    for i in range(boxes["class_labels"].shape[1]):
        max_idx = torch.argmax(boxes["class_labels"][0][i])
        obj_class = classes[max_idx]
        if size_half:
            x_min, x_max, y_min, y_max, z_min, z_max = (
                boxes["translations"][0][i][0] - boxes["sizes"][0][i][0],
                boxes["translations"][0][i][0] + boxes["sizes"][0][i][0],
                boxes["translations"][0][i][1] - boxes["sizes"][0][i][1],
                boxes["translations"][0][i][1] + boxes["sizes"][0][i][1],
                boxes["translations"][0][i][2] - boxes["sizes"][0][i][2],
                boxes["translations"][0][i][2] + boxes["sizes"][0][i][2],
            )
        else:
            x_min, x_max, y_min, y_max, z_min, z_max = (
                boxes["translations"][0][i][0] - boxes["sizes"][0][i][0] / 2,
                boxes["translations"][0][i][0] + boxes["sizes"][0][i][0] / 2,
                boxes["translations"][0][i][1] - boxes["sizes"][0][i][1] / 2,
                boxes["translations"][0][i][1] + boxes["sizes"][0][i][1] / 2,
                boxes["translations"][0][i][2] - boxes["sizes"][0][i][2] / 2,
                boxes["translations"][0][i][2] + boxes["sizes"][0][i][2] / 2,
            )

        draw_bounding_box_box(
            ax,
            (x_min, y_min, z_min, x_max, y_max, z_max),
            color=material_assets.get(obj_class, "black"),
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    print("image save to ", save_path)
    plt.close()


def draw_scene_list(
    boxes, classes, line=2, save_path="../draw.png", nms=None, size_half=False
):

    length = len(boxes)
    fig = plt.figure(figsize=(14, 12))
    col_num = int(length / line)
    for batch_i, item in enumerate(boxes):
        if nms:
            item = nms_and_merge(item)

        ax = fig.add_subplot(line, col_num, batch_i + 1, projection="3d")
        for i in range(item["class_labels"].shape[1]):
            max_idx = torch.argmax(item["class_labels"][0][i])
            obj_class = classes[max_idx]
            if size_half:
                x_min, x_max, y_min, y_max, z_min, z_max = (
                    item["translations"][0][i][0] - item["sizes"][0][i][0],
                    item["translations"][0][i][0] + item["sizes"][0][i][0],
                    item["translations"][0][i][1] - item["sizes"][0][i][1],
                    item["translations"][0][i][1] + item["sizes"][0][i][1],
                    item["translations"][0][i][2] - item["sizes"][0][i][2],
                    item["translations"][0][i][2] + item["sizes"][0][i][2],
                )
            else:
                x_min, x_max, y_min, y_max, z_min, z_max = (
                    item["translations"][0][i][0] - item["sizes"][0][i][0] / 2,
                    item["translations"][0][i][0] + item["sizes"][0][i][0] / 2,
                    item["translations"][0][i][1] - item["sizes"][0][i][1] / 2,
                    item["translations"][0][i][1] + item["sizes"][0][i][1] / 2,
                    item["translations"][0][i][2] - item["sizes"][0][i][2] / 2,
                    item["translations"][0][i][2] + item["sizes"][0][i][2] / 2,
                )

            # draw_bounding_box(ax, (x_min,y_min,z_min,x_max,y_max,z_max))
            # draw_bounding_box_line(ax, (x_min,y_min,z_min,x_max,y_max,z_max),color=material_assets.get(obj_class,'black'))
            draw_bounding_box_box(
                ax,
                (x_min, y_min, z_min, x_max, y_max, z_max),
                color=material_assets.get(obj_class, "black"),
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)

    handles = []
    for obj_class, color in material_assets.items():
        handles.append(
            plt.Line2D(
                [0], [0], marker="o", color="w", label=obj_class, markerfacecolor=color
            )
        )
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0, 0.95),
        bbox_transform=fig.transFigure,
    )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_json(boxes, classes, save_path="../draw.json", nms=None, size_half=False):

    if nms:
        boxes = nms_and_merge(boxes)
    building = []
    for i in range(boxes["class_labels"].shape[1]):
        max_idx = torch.argmax(boxes["class_labels"][0][i])
        obj_class = classes[max_idx] + "Material"
        size = boxes["sizes"][0][i].tolist()
        translation = boxes["translations"][0][i].tolist()

        building.append(
            {
                "actor_label": f"Cube{i}",
                "materials": [obj_class],
                "actor_size": size,
                "actor_location": translation,
            }
        )
    with open(save_path, "w") as f:
        json.dump(building, f)
    print("json save to ", save_path)


furniture_color = [
    (0.8031113829403007, 0.8559631555630488, 0.4632356993707173),
    (0.25489709749769085, 0.4786806290962444, 0.6199908044011786),
    (0.4417606730902278, 0.11673246190590914, 0.7369919456508808),
    (0.8275710482533588, 0.09562295812905242, 0.04888580884489779),
    (0.48906758723595734, 0.21175769401095146, 0.4814698755078809),
    (0.7812722599199563, 0.6075509239673018, 0.6230939422769749),
    (0.56299722353235, 0.7087397534988379, 0.5528337474599023),
    (0.08587994302250701, 0.7027955338656938, 0.03310159417878167),
    (0.32705071459986557, 0.5226992424213799, 0.3919853289813269),
    (0.1598903425966942, 0.6217785836594822, 0.09812377768069902),
    (0.6536642394728259, 0.040710085286695175, 0.1406936316671712),
    (0.566458813958767, 0.4802935087916391, 0.2151521202654728),
    (0.7401692739947016, 0.35621618864541094, 0.818826452194732),
    (0.8417401414017431, 0.3114243324073337, 0.9926082503945957),
    (0.47235112051914563, 0.598577636998914, 0.0937732066829382),
    (0.018174980007872987, 0.3064245256615228, 0.2930000651933553),
    (0.9506548415297471, 0.6881578020086272, 0.3395669859564405),
    (0.21156512732882093, 0.11633030393085242, 0.8876989333877642),
    (0.5752918655344652, 0.8086385219787382, 0.15968577827366492),
    (0.1997116214593032, 0.03703911543956406, 0.0020812501184519494),
    (0.5433983198399703, 0.026977288323892457, 0.03986692652588819),
    (0.5843344997949416, 0.8348494545702515, 0.9342993177577382),
    (0.43669608764029566, 0.23440825194347903, 0.7673761775811515),
    (0.7742194896924459, 0.1840484167062394, 0.9068149075107718),
    (0.23103985306106956, 0.45837959262702443, 0.8750049903095883),
    (0.148511338744036, 0.6596793279023146, 0.1808608715375818),
    (0.03719623120972504, 0.46573229703717367, 0.12633962935997756),
    (0.19185270228866458, 0.1236207327728509, 0.4184376438169104),
    (0.8433675569411822, 0.9541919379840698, 0.9356226980931329),
    (0.46812842000428767, 0.14710583912041075, 0.2628731315954884),
    (0.2847451332055282, 0.5546253126589816, 0.9723817426467891),
    (0.6271642128121215, 0.6300712895459625, 0.8412475481453898),
    (0.013707147181634238, 0.8166143225939565, 0.14120062341420614),
    (0.8222251500351221, 0.6675209261177865, 0.766904933963525),
    (0.7579531465530278, 0.2769861224197765, 0.9348026861642736),
    (0.34584309701733373, 0.0369599163923533, 0.3997572203334757),
    (0.18919912203891243, 0.42404097765787696, 0.7866758715447777),
    (0.6434584112920234, 0.9925121116246192, 0.12817340028833812),
    (0.6510209800399074, 0.30012501556510895, 0.6373814364960249),
    (0.03668586166392385, 0.4491787333555264, 0.7820395124098405),
    (0.08651026397333494, 0.641523667239412, 0.6146339504922707),
    (0.48816182807863284, 0.9489171909037752, 0.9031718580603779),
    (0.5255670510317563, 0.5331097794328499, 0.454600336534416),
    (0.8667405812132575, 0.9510641377622997, 0.13425655694802285),
    (0.279969692640365, 0.3595159603723117, 0.8602158192197014),
    (0.688432697777241, 0.8024634741319663, 0.9175256668479886),
    (0.6233068354194125, 0.02484450521659587, 0.6752416070486057),
    (0.7601795667817143, 0.003514725100770666, 0.8715188899781686),
    (0.146872516000084, 0.49784676908694925, 0.9916350351060752),
    (0.04314162104305319, 0.745091047199688, 0.7506565817440585),
]


def draw_scene_3dfront_list(
    boxes,
    classes,
    line=2,
    save_path="../draw.png",
    nms=None,
    size_half=False,
    changeYZ=True,
    top_view=False,
):

    length = len(boxes)
    fig = plt.figure(figsize=(14, 12))
    col_num = length / line
    for batch_i, item in enumerate(boxes):
        if nms:
            item = nms_and_merge(item)

        ax = fig.add_subplot(line, col_num, batch_i + 1, projection="3d")
        for i in range(item["class_labels"].shape[1]):
            max_idx = torch.argmax(item["class_labels"][0][i])
            obj_class = classes[max_idx]
            angles = item["angles"][0][i][0]

            if size_half:
                x_min, x_max, y_min, y_max, z_min, z_max = (
                    item["translations"][0][i][0] - item["sizes"][0][i][0],
                    item["translations"][0][i][0] + item["sizes"][0][i][0],
                    item["translations"][0][i][1] - item["sizes"][0][i][1],
                    item["translations"][0][i][1] + item["sizes"][0][i][1],
                    item["translations"][0][i][2] - item["sizes"][0][i][2],
                    item["translations"][0][i][2] + item["sizes"][0][i][2],
                )
            else:
                x_min, x_max, y_min, y_max, z_min, z_max = (
                    item["translations"][0][i][0] - item["sizes"][0][i][0] / 2,
                    item["translations"][0][i][0] + item["sizes"][0][i][0] / 2,
                    item["translations"][0][i][1] - item["sizes"][0][i][1] / 2,
                    item["translations"][0][i][1] + item["sizes"][0][i][1] / 2,
                    item["translations"][0][i][2] - item["sizes"][0][i][2] / 2,
                    item["translations"][0][i][2] + item["sizes"][0][i][2] / 2,
                )

            furniture_color = color_palette = np.array(
                sns.color_palette("hls", len(classes))
            )

            color = furniture_color[max_idx]

            draw_bounding_box_box(
                ax,
                (x_min, y_min, z_min, x_max, y_max, z_max),
                color=color,
                rotation_angle=angles,
                rotation_axis="y",
                changeYZ=changeYZ,
            )
        if top_view:
            ax.view_init(elev=90, azim=0)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_zlim(-5, 5)

    handles = []
    for obj_class, color in zip(classes, furniture_color):
        handles.append(
            plt.Line2D(
                [0], [0], marker="o", color="w", label=obj_class, markerfacecolor=color
            )
        )
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0, 0.95),
        bbox_transform=fig.transFigure,
    )

    plt.tight_layout()
    if top_view:
        save_path = save_path[:-4] + "_top" + ".png"
    plt.savefig(save_path)
    plt.close()


def main(argv):
    parser = argparse.ArgumentParser(
        description="Generate scenes using a previously trained model"
    )

    parser.add_argument(
        "config_file",
        help="Path to the file that contains the experiment configuration",
    )
    parser.add_argument(
        "--output_directory", default="/tmp/", help="Path to the output directory"
    )
    parser.add_argument(
        "--weight_file", default=None, help="Path to a pretrained model"
    )
    parser.add_argument(
        "--n_sequences",
        default=10,
        type=int,
        help="The number of layouts to be generated",
    )
    parser.add_argument("--clip_denoised", action="store_true", help="if clip_denoised")
    parser.add_argument("--fix_order", action="store_true", help="if use fix order")
    parser.add_argument(
        "--save_path", type=str, default="../sample", help="mesh format "
    )
    args = parser.parse_args(argv)

    # Disable trimesh's logger
    logging.getLogger("trimesh").setLevel(logging.ERROR)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Running code on", device)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    config = load_config(args.config_file)

    ########## make it for evaluation
    if "text" in config["data"]["encoding_type"]:
        if "textfix" not in config["data"]["encoding_type"]:
            config["data"]["encoding_type"] = config["data"]["encoding_type"].replace(
                "text", "textfix"
            )

    if "no_prm" not in config["data"]["encoding_type"]:
        print("NO PERM AUG in test")
        config["data"]["encoding_type"] = config["data"]["encoding_type"] + "_no_prm"
    print("encoding type :", config["data"]["encoding_type"])
    #######

    raw_dataset, train_dataset = get_dataset_raw_and_encoded(
        config["data"],
        filter_fn=filter_function(
            config["data"], split=config["training"].get("splits", ["train", "val"])
        ),
        split=config["training"].get("splits", ["train", "val"]),
    )

    raw_dataset, dataset = get_dataset_raw_and_encoded(
        config["data"],
        filter_fn=filter_function(
            config["data"], split=config["validation"].get("splits", ["test"])
        ),
        split=config["validation"].get("splits", ["test"]),
    )
    print(
        "Loaded {} scenes with {} object types:".format(
            len(dataset), dataset.n_object_types
        )
    )
    network, _, _ = build_network(
        dataset.feature_size, dataset.n_classes, config, args.weight_file, device=device
    )
    network.eval()

    size_half = config["network"].get("size_half", False)

    classes = np.array(dataset.class_labels)
    print("class labels:", classes, len(classes))
    for i in range(args.n_sequences):

        seperate = True
        val_batch = 16
        bbox_params = network.generate_layout(
            room_mask=torch.zeros([val_batch, 64]).to(device),
            batch_size=val_batch,
            num_points=config["network"]["sample_num_points"],
            point_dim=config["network"]["point_dim"],
            text=["This is a building."] * val_batch,
            device=device,
            clip_denoised=args.clip_denoised,
            batch_seeds=torch.arange(i, i + 1),
        )

        boxes = (
            [dataset.post_process(bbox_param) for bbox_param in bbox_params]
            if isinstance(bbox_params, list)
            else dataset.post_process(bbox_params)
        )
        os.makedirs(os.path.join(args.save_path, "raw"), exist_ok=True)
        os.makedirs(os.path.join(args.save_path, "merge"), exist_ok=True)

        for bat_id in range(len(boxes)):
            save_json(
                boxes[bat_id],
                classes,
                save_path=os.path.join(args.save_path, "raw", f"{i}_{bat_id}.json"),
            )
            save_json(
                boxes[bat_id],
                classes,
                save_path=os.path.join(
                    args.save_path, "merge", f"nms{i}_{bat_id}.json"
                ),
                nms=True,
            )

        if isinstance(bbox_params, list) and not (seperate):
            draw_scene_list(
                boxes,
                classes,
                save_path=os.path.join(args.save_path, "raw", f"{i}.png"),
                size_half=size_half,
            )
            draw_scene_list(
                boxes,
                classes,
                save_path=os.path.join(args.save_path, "merge", f"nms{i}.png"),
                nms=True,
                size_half=size_half,
            )
        else:
            if isinstance(bbox_params, list):
                for bat_id in range(len(bbox_params)):
                    draw_scene(
                        boxes[bat_id],
                        classes,
                        save_path=os.path.join(
                            args.save_path, "raw", f"{i}_{bat_id}.png"
                        ),
                    )
                    draw_scene(
                        boxes[bat_id],
                        classes,
                        save_path=os.path.join(
                            args.save_path, "merge", f"nms{i}_{bat_id}.png"
                        ),
                        nms=True,
                    )
            else:
                draw_scene(
                    boxes,
                    classes,
                    save_path=os.path.join(args.save_path, "raw", f"{i}.png"),
                )
                draw_scene(
                    boxes,
                    classes,
                    save_path=os.path.join(args.save_path, "merge", f"nms{i}.png"),
                    nms=True,
                )


if __name__ == "__main__":
    main(sys.argv[1:])
