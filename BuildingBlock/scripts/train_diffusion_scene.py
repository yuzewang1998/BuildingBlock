#
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
#

"""Script used to train a ATISS."""
import argparse
import logging
from multiprocessing import Manager
import os
import shutil
import sys
import time

import numpy as np

import swanlab
import torch
from torch.utils.data import DataLoader
import wandb

from generate_diffusion_building import draw_scene_3dfront_list
from training_utils import (
    id_generator,
    save_experiment_params,
    load_config,
    yield_forever,
    load_checkpoints,
    save_checkpoints,
)

from scene_synthesis.datasets import get_encoded_dataset, filter_function
from scene_synthesis.networks import (
    build_network,
    optimizer_factory,
    schedule_factory,
    adjust_learning_rate,
)
from scene_synthesis.stats_logger import StatsLogger, WandB, Swanlab

from scene_synthesis.datasets import filter_function, get_dataset_raw_and_encoded
from scene_synthesis.datasets.threed_front import ThreedFront
from scene_synthesis.datasets.threed_future_dataset import ThreedFutureDataset
from scene_synthesis.networks import build_network
from scene_synthesis.utils import (
    get_textured_objects,
    get_textured_objects_based_on_objfeats,
)
from scene_synthesis.stats_logger import AverageAggregator
from utils import floor_plan_from_scene, export_scene, get_textured_objects_in_scene
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
from cleanfid import fid
from copy import deepcopy
from PIL import Image


# Build the dataset of 3D models
objects_dataset = None

raw_dataset, dataset = None, None


def load_dataset(args, config):
    global objects_dataset, raw_dataset, dataset
    # Build the dataset of 3D models
    objects_dataset = ThreedFutureDataset.from_pickled_dataset(
        args.path_to_pickled_3d_futute_models
    )
    print("Loaded {} 3D-FUTURE models".format(len(objects_dataset)))

    raw_dataset, dataset = get_dataset_raw_and_encoded(
        config["data"],
        filter_fn=filter_function(
            config["data"], split=config["validation"].get("splits", ["test"])
        ),
        split=config["validation"].get("splits", ["test"]),
        data_cache={},
    )
    print(
        "Loaded {} scenes with {} object types:".format(
            len(dataset), dataset.n_object_types
        )
    )


def eval(args, config, network):
    if args.render_top2down:
        if args.without_floor:
            scene_top2down = Scene(size=(256, 256), background=[1, 1, 1, 1])
        else:
            scene_top2down = Scene(size=(256, 256), background=[0, 0, 0, 1])
        scene_top2down.up_vector = (0, 0, -1)
        scene_top2down.camera_target = (0, 0, 0)
        scene_top2down.camera_position = (0, 4, 0)
        scene_top2down.light = (0, 4, 0)
        scene_top2down.camera_matrix = Matrix44.orthogonal_projection(
            left=-3.1, right=3.1, bottom=3.1, top=-3.1, near=0.1, far=6
        )

    # print('init scene top2donw')
    given_scene_id = None
    if args.scene_id:
        for i, di in enumerate(raw_dataset):
            if str(di.scene_id) == args.scene_id:
                given_scene_id = i

    if args.compute_intersec:
        num_objects_counter = []
        total_num_symmetry, total_num_pairs = 0, 0
        NUM_OBJ = AverageAggregator()
        NUM_PAIRS = AverageAggregator()
        BOX_IOU = AverageAggregator()
        BOX_INSEC = AverageAggregator()
        OVERLAP_RATIO = AverageAggregator()

        num_objects_counter_onlysize = []
        total_num_symmetry_onlysize, total_num_pairs_onlysize = 0, 0
        NUM_OBJ_ONLYSIZE = AverageAggregator()
        NUM_PAIRS_ONLYSIZE = AverageAggregator()
        BOX_IOU_ONLYSIZE = AverageAggregator()
        BOX_INSEC_ONLYSIZE = AverageAggregator()
        OVERLAP_RATIO_ONLYSIZE = AverageAggregator()

    classes = np.array(dataset.class_labels)
    # print('class labels:', classes, len(classes))
    val_batch = config["validation"].get("batch_size", 1)
    # for b, samples_batch in zip(range(args.n_sequences//val_batch), yield_forever(data_loader)):
    batch_sample = []
    batch_scene = []
    iList = []
    scene_idx_list = []
    for i in range(args.n_sequences):
        if args.fix_order:
            if i < len(dataset):
                scene_idx = given_scene_id or i
            else:
                scene_idx = given_scene_id or (i % len(dataset))
        else:
            scene_idx = given_scene_id or np.random.choice(len(dataset))

        current_scene = raw_dataset[scene_idx]
        samples = dataset[scene_idx]
        # print("{} / {}: Using the {} floor plan of scene {}".format(
        #     i, args.n_sequences, scene_idx, current_scene.scene_id)
        # )
        # Get a floor plan

        if len(batch_sample) < val_batch and i != args.n_sequences - 1:
            batch_sample.append(deepcopy(samples))
            batch_scene.append(deepcopy(current_scene))
            iList.append(i)
            scene_idx_list.append(scene_idx)
            continue
        else:
            # print("batchsize now:",len(batch_sample))
            floor_plans, tr_floors, room_masks, texts = [], [], [], []
            for samples, current_scene in zip(batch_sample, batch_scene):
                floor_plan, tr_floor, room_mask = floor_plan_from_scene(
                    current_scene,
                    args.path_to_floor_plan_textures,
                    no_texture=args.no_texture,
                )
                floor_plans.append(floor_plan)
                tr_floors.append(tr_floor)
                room_masks.append(room_mask)
                if "description" in samples.keys():
                    texts.append(samples["description"])

            room_masks = torch.cat(room_masks, 0)

            # print("room_masks: ",room_masks)

        # import ipdb;ipdb.set_trace()
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        # import ipdb;ipdb.set_trace()
        bbox_params = network.generate_layout(
            # room_mask=room_mask.to(device),
            room_mask=room_masks.to(device),
            # batch_size=val_batch if not config['network'].get('text_condition',None) else len(texts),
            batch_size=len(iList),
            num_points=config["network"]["sample_num_points"],
            point_dim=config["network"]["point_dim"],
            # text=torch.from_numpy(samples['desc_emb'])[None, :].to(device) if 'desc_emb' in samples.keys() else None, # glove embedding
            # text=samples['description'] if 'description' in samples.keys() else None,  # bert
            text=texts,
            device=device,
            clip_denoised=args.clip_denoised,
            batch_seeds=torch.arange(i, i + 1),
        )

        # boxes = dataset.post_process(bbox_params)
        boxeses = [dataset.post_process(bbox_param) for bbox_param in bbox_params]

        for samples, current_scene, boxes, it, scene_idx in zip(
            batch_sample, batch_scene, boxeses, iList, scene_idx_list
        ):
            bbox_params_t = (
                torch.cat(
                    [
                        boxes["class_labels"],
                        boxes["translations"],
                        boxes["sizes"],
                        boxes["angles"],
                    ],
                    dim=-1,
                )
                .cpu()
                .numpy()
            )
            # print('Generated bbox:', bbox_params_t.shape)

            try:
                if args.retrive_objfeats:
                    objfeats = boxes["objfeats"].cpu().numpy()
                    # print('shape retrieval based on obj latent feats')

                    renderables, trimesh_meshes, model_jids = (
                        get_textured_objects_based_on_objfeats(
                            bbox_params_t,
                            objects_dataset,
                            classes,
                            diffusion=True,
                            no_texture=args.no_texture,
                            query_objfeats=objfeats,
                        )
                    )
                    (
                        renderables_onlysize,
                        trimesh_meshes_onlysize,
                        model_jids_onlysize,
                    ) = get_textured_objects(
                        bbox_params_t,
                        objects_dataset,
                        classes,
                        diffusion=True,
                        no_texture=args.no_texture,
                    )
                else:
                    renderables, trimesh_meshes, model_jids = get_textured_objects(
                        bbox_params_t,
                        objects_dataset,
                        classes,
                        diffusion=True,
                        no_texture=args.no_texture,
                    )

                if not args.without_floor:
                    renderables += floor_plan
                    trimesh_meshes += tr_floor

                if args.render_top2down:
                    path_to_image = "{}/{}/{}/{}_{}_{:03d}.png".format(
                        args.output_directory,
                        "tmp",
                        "result",
                        current_scene.scene_id,
                        scene_idx,
                        it,
                    )
                    os.makedirs(os.path.dirname(path_to_image), exist_ok=True)
                    render_top2down(
                        scene_top2down,
                        renderables,
                        color=None,
                        mode="shading",
                        frame_path=path_to_image,
                    )

                    if args.retrive_objfeats:
                        # save results of only retrieving sizes
                        # path_to_image_onlysize = "{}/{}".format(
                        #     args.output_directory,
                        #     "retrive_only_size",
                        # )
                        # if not os.path.exists(path_to_image_onlysize):
                        #     os.mkdir(path_to_image_onlysize)
                        path_to_image_onlysize = "{}/{}/{}/{}_{}_{:03d}.png".format(
                            args.output_directory,
                            "tmp",
                            "retrive_only_size",
                            current_scene.scene_id,
                            scene_idx,
                            it,
                        )
                        os.makedirs(
                            os.path.dirname(path_to_image_onlysize), exist_ok=True
                        )
                        render_top2down(
                            scene_top2down,
                            renderables_onlysize,
                            color=None,
                            mode="shading",
                            frame_path=path_to_image_onlysize,
                        )

                    if args.compute_intersec:
                        num_objects, num_pairs, avg_iou, avg_insec, overlap_ratio = (
                            computer_intersection(trimesh_meshes)
                        )
                        num_objects_counter.append(num_objects)
                        NUM_OBJ.value = num_objects
                        NUM_PAIRS.value = num_pairs
                        BOX_IOU.value = avg_iou
                        BOX_INSEC.value = avg_insec
                        OVERLAP_RATIO.value = overlap_ratio
                        num_symmetries = computer_symmetry(
                            trimesh_meshes,
                            boxes["class_labels"][0, :, :].cpu().numpy(),
                            model_jids,
                        )
                        total_num_symmetry += num_symmetries
                        total_num_pairs += num_pairs
                        string = "num scenes: {:d} - num objects avg: {:f} - std: {:f} - num pairs: {:f} - box iou: {:f} - box intersec: {:f} - overlap ratio: {:f} - total num symmetries: {:d} - total num pairs: {:d} retrive objfeats  ".format(
                            it + 1,
                            NUM_OBJ.value,
                            np.array(num_objects_counter).std(),
                            NUM_PAIRS.value,
                            BOX_IOU.value,
                            BOX_INSEC.value,
                            OVERLAP_RATIO.value,
                            total_num_symmetry,
                            total_num_pairs,
                        )
                        print(string)
                        with open(
                            os.path.join(args.output_directory, "iou_states.txt"), "a"
                        ) as f:
                            f.write(string + "\n")
                        f.close()

                        if args.retrive_objfeats:
                            (
                                num_objects,
                                num_pairs,
                                avg_iou,
                                avg_insec,
                                overlap_ratio,
                            ) = computer_intersection(trimesh_meshes_onlysize)
                            num_objects_counter_onlysize.append(num_objects)
                            NUM_OBJ_ONLYSIZE.value = num_objects
                            NUM_PAIRS_ONLYSIZE.value = num_pairs
                            BOX_IOU_ONLYSIZE.value = avg_iou
                            BOX_INSEC_ONLYSIZE.value = avg_insec
                            OVERLAP_RATIO_ONLYSIZE.value = overlap_ratio

                            num_symmetries = computer_symmetry(
                                trimesh_meshes_onlysize,
                                boxes["class_labels"][0, :, :].cpu().numpy(),
                                model_jids_onlysize,
                            )
                            total_num_symmetry_onlysize += num_symmetries
                            total_num_pairs_onlysize += num_pairs
                            string = "num scenes: {:d} - num objects avg: {:f} - std: {:f} - num pairs: {:f} - box iou: {:f} - box intersec: {:f} - overlap ratio: {:f} - total num symmetries: {:d} - total num pairs: {:d} retrive only size".format(
                                it + 1,
                                NUM_OBJ_ONLYSIZE.value,
                                np.array(num_objects_counter_onlysize).std(),
                                NUM_PAIRS_ONLYSIZE.value,
                                BOX_IOU_ONLYSIZE.value,
                                BOX_INSEC_ONLYSIZE.value,
                                OVERLAP_RATIO_ONLYSIZE.value,
                                total_num_symmetry_onlysize,
                                total_num_pairs_onlysize,
                            )
                            print(string)
                            with open(
                                os.path.join(args.output_directory, "iou_states.txt"),
                                "a",
                            ) as f:
                                f.write(string + "\n")
                            f.close()

                if args.save_mesh:
                    if trimesh_meshes is not None:
                        # Create a trimesh scene and export it
                        path_to_objs = os.path.join(
                            args.output_directory,
                            "scene_mesh",
                        )
                        if not os.path.exists(path_to_objs):
                            os.mkdir(path_to_objs)
                        filename = "{}_{}_{:03d}".format(
                            current_scene.scene_id, scene_idx, it
                        )
                        path_to_scene = os.path.join(
                            path_to_objs, filename + args.mesh_format
                        )
                        whole_scene_mesh = merge_meshes(trimesh_meshes)
                        o3d.io.write_triangle_mesh(path_to_scene, whole_scene_mesh)

                    if args.retrive_objfeats:
                        if trimesh_meshes_onlysize is not None:
                            # Create a trimesh scene and export it
                            path_to_objs_retrive_onlysize = os.path.join(
                                args.output_directory,
                                "scene_mesh_retrive_onlysize",
                            )
                            if not os.path.exists(path_to_objs_retrive_onlysize):
                                os.mkdir(path_to_objs_retrive_onlysize)

                            filename = "{}_{}_{:03d}".format(
                                current_scene.scene_id, scene_idx, it
                            )
                            path_to_scene_onlysize = os.path.join(
                                path_to_objs_retrive_onlysize,
                                filename + args.mesh_format,
                            )
                            whole_scene_mesh_onlysize = merge_meshes(
                                trimesh_meshes_onlysize
                            )
                            o3d.io.write_triangle_mesh(
                                path_to_scene_onlysize, whole_scene_mesh_onlysize
                            )

                if "description" in samples.keys():
                    path_to_texts = os.path.join(
                        args.output_directory,
                        "{}_{}_{:03d}_text.txt".format(
                            current_scene.scene_id, scene_idx, it
                        ),
                    )
                    # print('the length of samples[description]: {:d}'.format( len(samples['description']) ) )
                    # print('text description {}'.format( samples['description']) )
                    open(path_to_texts, "w").write("".join(samples["description"]))
            except:
                pass
        batch_sample, batch_scene, boxeses, iList, scene_idx_list = [], [], [], [], []


def merge_images_from_path(image_dir, save_path, num_images=4):

    image_files = [
        f
        for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    image_files.sort()
    image_files = image_files[:num_images]

    images = []
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        img = Image.open(img_path).convert("RGB")
        images.append(img)

    if not images:
        print(f"No images found in {image_dir}. Skipping...")
        return

    target_size = (256, 256)
    resized_images = [img.resize(target_size) for img in images]

    total_width = target_size[0] * len(resized_images)
    max_height = target_size[1]

    merged_image = Image.new("RGB", (total_width, max_height))

    x_offset = 0
    for img in resized_images:
        merged_image.paste(img, (x_offset, 0))
        x_offset += img.width

    merged_image.save(save_path)
    # print(f"Merged image saved to {save_path}")


def merge_images_2x2(image_dir, save_path, num_images=4):

    image_files = [
        f
        for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    image_files.sort()
    image_files = image_files[:num_images]

    images = []
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        img = Image.open(img_path).convert("RGB")
        images.append(img)

    if not images:
        print(f"No images found in {image_dir}. Skipping...")
        return

    target_size = (256, 256)
    resized_images = [img.resize(target_size) for img in images]

    grid_width = target_size[0] * 2
    grid_height = target_size[1] * ((len(resized_images) + 1) // 2)

    merged_image = Image.new("RGB", (grid_width, grid_height))

    for idx, img in enumerate(resized_images):
        row = idx // 2
        col = idx % 2
        x_offset = col * target_size[0]
        y_offset = row * target_size[1]
        merged_image.paste(img, (x_offset, y_offset))

    merged_image.save(save_path)


def main(argv):
    parser = argparse.ArgumentParser(
        description="Train a generative model on bounding boxes"
    )

    parser.add_argument(
        "config_file",
        help="Path to the file that contains the experiment configuration",
    )
    parser.add_argument("output_directory", help="Path to the output directory")
    parser.add_argument(
        "path_to_pickled_3d_futute_models", help="Path to the 3D-FUTURE model meshes"
    )
    parser.add_argument(
        "--weight_file",
        default=None,
        help=(
            "The path to a previously trained model to continue" " the training from"
        ),
    )
    parser.add_argument(
        "--continue_from_epoch",
        default=0,
        type=int,
        help="Continue training from epoch (default=0)",
    )
    parser.add_argument(
        "--n_processes",
        type=int,
        default=0,
        help="The number of processed spawned by the batch provider",
    )
    parser.add_argument("--seed", type=int, default=27, help="Seed for the PRNG")
    parser.add_argument(
        "--experiment_tag",
        default=None,
        help="Tag that refers to the current experiment",
    )
    parser.add_argument(
        "--with_wandb_logger",
        action="store_true",
        help="Use wandB for logging the training progress",
    )
    parser.add_argument(
        "--with_swanlab_logger",
        action="store_true",
        help="Use wandB for logging the training progress",
    )
    parser.add_argument("--render_eval", action="store_true", help="render eval or not")
    parser.add_argument(
        "--path_to_floor_plan_textures",
        default="../demo/floor_plan_texture_images",
        help="Path to floor texture images",
    )
    parser.add_argument(
        "--n_sequences",
        default=10,
        type=int,
        help="The number of sequences to be generated",
    )
    parser.add_argument(
        "--background",
        type=lambda x: list(map(float, x.split(","))),
        default="1,1,1,1",
        help="Set the background of the scene",
    )
    parser.add_argument(
        "--up_vector",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,1,0",
        help="Up vector of the scene",
    )
    parser.add_argument(
        "--camera_position",
        type=lambda x: tuple(map(float, x.split(","))),
        default="-0.10923499,1.9325259,-7.19009",
        help="Camer position in the scene",
    )
    parser.add_argument(
        "--camera_target",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,0,0",
        help="Set the target for the camera",
    )
    parser.add_argument(
        "--window_size",
        type=lambda x: tuple(map(int, x.split(","))),
        default="512,512",
        help="Define the size of the scene and the window",
    )
    parser.add_argument(
        "--with_rotating_camera",
        action="store_true",
        help="Use a camera rotating around the object",
    )
    parser.add_argument(
        "--save_frames", help="Path to save the visualization frames to"
    )
    parser.add_argument(
        "--n_frames", type=int, default=360, help="Number of frames to be rendered"
    )
    parser.add_argument(
        "--without_screen", action="store_true", help="Perform no screen rendering"
    )
    parser.add_argument(
        "--scene_id", default=None, help="The scene id to be used for conditioning"
    )
    parser.add_argument(
        "--render_top2down",
        action="store_true",
        help="Perform top2down orthographic rendering",
    )
    parser.add_argument(
        "--without_floor", action="store_true", help="if remove the floor plane"
    )
    parser.add_argument(
        "--no_texture", action="store_true", help="if remove the texture"
    )
    parser.add_argument("--save_mesh", action="store_true", help="if save mesh")
    parser.add_argument("--mesh_format", type=str, default=".ply", help="mesh format ")
    parser.add_argument("--clip_denoised", action="store_true", help="if clip_denoised")
    #
    parser.add_argument(
        "--retrive_objfeats",
        action="store_true",
        help="if retrive most similar objectfeats",
    )
    parser.add_argument("--fix_order", action="store_true", help="if use fix order")
    parser.add_argument(
        "--compute_intersec", action="store_true", help="if remove the texture"
    )

    args = parser.parse_args(argv)

    # Disable trimesh's logger
    logging.getLogger("trimesh").setLevel(logging.ERROR)

    # Set the random seed
    np.random.seed(args.seed)
    torch.manual_seed(np.random.randint(np.iinfo(np.int32).max))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(np.random.randint(np.iinfo(np.int32).max))

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Running code on", device)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    # Create an experiment directory using the experiment_tag
    if args.experiment_tag is None:
        experiment_tag = id_generator(9)
    else:
        experiment_tag = args.experiment_tag

    experiment_directory = os.path.join(args.output_directory, experiment_tag)
    if not os.path.exists(experiment_directory):
        os.makedirs(experiment_directory)

    # Save the parameters of this run to a file
    save_experiment_params(args, experiment_tag, experiment_directory)
    try:
        shutil.copy(args.config_file, os.path.join(experiment_directory, "config.yaml"))
    except shutil.SameFileError:
        print("The config file is the same as the one in the experiment directory")
    print("Save experiment statistics in {}".format(experiment_directory))

    # Parse the config file
    config = load_config(args.config_file)

    load_dataset(args, config)

    manager = Manager()
    train_data_cache = manager.dict()
    val_data_cache = manager.dict()
    train_dataset = get_encoded_dataset(
        config["data"],
        filter_function(
            config["data"], split=config["training"].get("splits", ["train", "val"])
        ),
        path_to_bounds=None,
        augmentations=config["data"].get("augmentations", None),
        split=config["training"].get("splits", ["train", "val"]),
        data_cache=train_data_cache,
    )

    # Build the dataset of 3D models
    objects_dataset = ThreedFutureDataset.from_pickled_dataset(
        args.path_to_pickled_3d_futute_models
    )
    print("Loaded {} 3D-FUTURE models".format(len(objects_dataset)))

    raw_dataset, dataset = get_dataset_raw_and_encoded(
        config["data"],
        filter_fn=filter_function(
            config["data"], split=config["validation"].get("splits", ["test"])
        ),
        split=config["validation"].get("splits", ["test"]),
        data_cache={},
    )
    print(
        "Loaded {} scenes with {} object types:".format(
            len(dataset), dataset.n_object_types
        )
    )
    classes = np.array(dataset.class_labels)

    # Compute the bounds for this experiment, save them to a file in the
    # experiment directory and pass them to the validation dataset
    path_to_bounds = os.path.join(experiment_directory, "bounds.npz")
    np.savez(
        path_to_bounds,
        sizes=train_dataset.bounds["sizes"],
        translations=train_dataset.bounds["translations"],
        angles=train_dataset.bounds["angles"],
        # add objfeats
        objfeats=train_dataset.bounds["objfeats"],
    )
    print("Saved the dataset bounds in {}".format(path_to_bounds))

    validation_dataset = get_encoded_dataset(
        config["data"],
        filter_function(
            config["data"], split=config["validation"].get("splits", ["test"])
        ),
        path_to_bounds=path_to_bounds,
        augmentations=None,
        split=config["validation"].get("splits", ["test"]),
        data_cache=val_data_cache,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"].get("batch_size", 128),
        num_workers=args.n_processes,
        collate_fn=train_dataset.collate_fn,
        shuffle=True,
        pin_memory=True,
        prefetch_factor=2,
    )
    print(
        "Loaded {} training scenes with {} object types".format(
            len(train_dataset), train_dataset.n_object_types
        )
    )
    print("Training set has {} bounds".format(train_dataset.bounds))

    val_loader = DataLoader(
        validation_dataset,
        batch_size=config["validation"].get("batch_size", 1),
        num_workers=args.n_processes,
        collate_fn=validation_dataset.collate_fn,
        shuffle=False,
    )
    print(
        "Loaded {} validation scenes with {} object types".format(
            len(validation_dataset), validation_dataset.n_object_types
        )
    )
    print("Validation set has {} bounds".format(validation_dataset.bounds))

    # Make sure that the train_dataset and the validation_dataset have the same
    # number of object categories
    assert train_dataset.object_types == validation_dataset.object_types

    # Build the network architecture to be used for training
    network, train_on_batch, validate_on_batch = build_network(
        train_dataset.feature_size,
        train_dataset.n_classes,
        config,
        args.weight_file,
        device=device,
    )
    n_all_params = int(sum([np.prod(p.size()) for p in network.parameters()]))
    n_trainable_params = int(
        sum(
            [
                np.prod(p.size())
                for p in filter(lambda p: p.requires_grad, network.parameters())
            ]
        )
    )
    print(
        f"Number of parameters in {network.__class__.__name__}:  {n_trainable_params} / {n_all_params}"
    )

    # Build an optimizer object to compute the gradients of the parameters
    optimizer = optimizer_factory(
        config["training"], filter(lambda p: p.requires_grad, network.parameters())
    )
    # optimizer = optimizer_factory(config["training"], network.parameters() )

    # Load the checkpoints if they exist in the experiment directory
    load_checkpoints(network, optimizer, experiment_directory, args, device)
    # Load the learning rate scheduler
    lr_scheduler = schedule_factory(config["training"])

    # Initialize the logger
    if args.with_wandb_logger:
        WandB.instance().init(
            config,
            model=network,
            project=config["logger"].get("project", "autoregressive_transformer"),
            name=experiment_tag,
            watch=False,
            log_frequency=10,
        )
    elif args.with_swanlab_logger:
        Swanlab.instance().init(
            config,
            model=network,
            project=config["logger"].get("project", "autoregressive_transformer"),
            name=experiment_tag,
            watch=False,
            log_frequency=10,
        )

    # Log the stats to a file
    StatsLogger.instance().add_output_file(
        open(os.path.join(experiment_directory, "stats.txt"), "w")
    )

    epochs = config["training"].get("epochs", 150) + 1
    steps_per_epoch = config["training"].get("steps_per_epoch", 500)
    save_every = config["training"].get("save_frequency", 10)
    val_every = config["validation"].get("frequency", 100)
    size_half = config["network"].get("size_half", True)

    # Do the training
    for i in range(args.continue_from_epoch, epochs):
        # adjust learning rate
        adjust_learning_rate(lr_scheduler, optimizer, i)

        network.train()
        batch_time_ema = 0
        batch_start_time = time.time()  # Start timing the batch
        # for b, sample in zip(range(steps_per_epoch), yield_forever(train_loader)):
        for b, sample in enumerate(train_loader):

            # Move everything to device
            # print(sample)
            for k, v in sample.items():
                if not isinstance(v, list):
                    sample[k] = v.to(device)
            network_start_time = time.time()  # Start timing the batch
            batch_loss = train_on_batch(network, optimizer, sample, config)

            batch_time_now = time.time() - batch_start_time  # Calculate batch time
            batch_start_time = time.time()
            # batch_time_now = time.time() - network_start_time  # Calculate batch time

            if batch_time_ema == 0:
                batch_time_ema = batch_time_now
            batch_time_ema = batch_time_ema * 0.9 + batch_time_now * 0.1
            # StatsLogger.instance().print_progress(i+1, b+1, batch_loss, batch_time_ema)
            StatsLogger.instance().print_progress(
                i + 1, b + 1, batch_loss, time_taken=batch_time_ema
            )

        if (i % save_every) == 0 and i != 0:
            save_checkpoints(
                i,
                network,
                optimizer,
                experiment_directory,
            )
        StatsLogger.instance().clear(step=i)

        # if i % val_every == 0 and i > 0:
        #     print("====> Validation Epoch ====>")
        #     network.eval()
        #     for b, sample in enumerate(val_loader):
        #         # Move everything to device
        #         for k, v in sample.items():
        #             if not isinstance(v, list):
        #                 sample[k] = v.to(device)
        #         batch_loss = validate_on_batch(network, sample, config)
        #         StatsLogger.instance().print_progress(-1, b+1, batch_loss)
        #     StatsLogger.instance().clear()
        #     print("====> Validation Epoch ====>")

        if i % val_every == 0 and i > 0:
            print("====> Validation Epoch ====>")
            network.eval()

            if not args.render_eval:
                val_batch = 4
                bbox_params = network.generate_layout(
                    room_mask=torch.zeros([val_batch, 64]).to(device),
                    batch_size=val_batch,
                    num_points=config["network"]["sample_num_points"],
                    point_dim=config["network"]["point_dim"],
                    # text=torch.from_numpy(samples['desc_emb'])[None, :].to(device) if 'desc_emb' in samples.keys() else None, # glove embedding
                    # text=samples['description'] if 'description' in samples.keys() else None,  # bert
                    text="",
                    device=device,
                    clip_denoised=True,
                    batch_seeds=torch.arange(0, 1),
                )
                classes = np.array(validation_dataset.class_labels)

                boxes = [
                    validation_dataset.post_process(bbox_param)
                    for bbox_param in bbox_params
                ]
                draw_scene_3dfront_list(
                    boxes,
                    classes,
                    save_path=os.path.join(
                        args.output_directory, "valEpoch" + str(i) + ".png"
                    ),
                    size_half=size_half,
                    top_view=False,
                )
                draw_scene_3dfront_list(
                    boxes,
                    classes,
                    save_path=os.path.join(
                        args.output_directory, "valEpoch" + str(i) + ".png"
                    ),
                    size_half=size_half,
                    top_view=True,
                )
                StatsLogger.instance().log(
                    {
                        "images/side_view": os.path.join(
                            args.output_directory, "valEpoch" + str(i) + ".png"
                        ),
                        "images/top_view": os.path.join(
                            args.output_directory, "valEpoch" + str(i) + ".png"
                        )[:-4]
                        + "_top"
                        + ".png",
                    },
                    step=i,
                )

            else:
                try:
                    shutil.rmtree(os.path.join(args.output_directory, "tmp"))
                except:
                    pass
                eval(args, config, network)
                file_path_real = "../sample_GT/sample_bedroom"
                fid_ours = fid.compute_fid(
                    file_path_real,
                    os.path.join(args.output_directory, "tmp", "result"),
                    device=torch.device("cuda"),
                )
                kid_ours = fid.compute_kid(
                    file_path_real,
                    os.path.join(args.output_directory, "tmp", "result"),
                    device=torch.device("cuda"),
                )
                save_path_ours = os.path.join(
                    args.output_directory, f"valEpoch{i}_result.png"
                )
                merge_images_2x2(
                    os.path.join(args.output_directory, "tmp", "result"), save_path_ours
                )

                if os.path.exists(
                    os.path.join(args.output_directory, "tmp", "retrive_only_size")
                ):
                    fid_size = fid.compute_fid(
                        file_path_real,
                        os.path.join(args.output_directory, "tmp", "retrive_only_size"),
                        device=torch.device("cuda"),
                    )
                    kid_size = fid.compute_kid(
                        file_path_real,
                        os.path.join(args.output_directory, "tmp", "retrive_only_size"),
                        device=torch.device("cuda"),
                    )
                    save_path_size = os.path.join(
                        args.output_directory, f"valEpoch{i}_retrive_only_size.png"
                    )
                    merge_images_2x2(
                        os.path.join(args.output_directory, "tmp", "retrive_only_size"),
                        save_path_size,
                    )

                if args.with_wandb_logger or args.with_swanlab_logger:
                    # import ipdb;ipdb.set_trace()
                    image_ours = Image.open(save_path_ours)
                    image_size = (
                        Image.open(save_path_size)
                        if os.path.exists(
                            os.path.join(
                                args.output_directory, "tmp", "retrive_only_size"
                            )
                        )
                        else image_ours
                    )
                    if args.with_wandb_logger:
                        image_ours = wandb.Image(image_ours, caption="ours")
                        image_size = wandb.Image(image_size, caption="only_size")
                    elif args.with_swanlab_logger:
                        image_ours = swanlab.Image(image_ours, caption="ours")
                        image_size = swanlab.Image(image_size, caption="only_size")
                    StatsLogger.instance().log(
                        {
                            "metrics/fid_ours": fid_ours,
                            "metrics/fid_size": (
                                fid_size
                                if os.path.exists(
                                    os.path.join(
                                        args.output_directory,
                                        "tmp",
                                        "retrive_only_size",
                                    )
                                )
                                else fid_ours
                            ),
                            "metrics/kid_ours": kid_ours,
                            "metrics/kid_size": (
                                kid_size
                                if os.path.exists(
                                    os.path.join(
                                        args.output_directory,
                                        "tmp",
                                        "retrive_only_size",
                                    )
                                )
                                else kid_ours
                            ),
                            "images/image_ours": image_ours,
                            "images/image_size": image_size,
                        },
                        step=i,
                    )

                # StatsLogger.log({

                # })
            for b, sample in enumerate(val_loader):
                # Move everything to device
                for k, v in sample.items():
                    if not isinstance(v, list):
                        sample[k] = v.to(device)
                batch_loss = validate_on_batch(network, sample, config)
                StatsLogger.instance().print_progress(-1, b + 1, batch_loss)
            StatsLogger.instance().clear(step=i)
            print("====> Validation Epoch ====>")


if __name__ == "__main__":
    main(sys.argv[1:])
