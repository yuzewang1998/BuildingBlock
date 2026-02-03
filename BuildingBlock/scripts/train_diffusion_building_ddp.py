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
import os
import shutil
import sys
import time
import numpy as np
from multiprocessing import Manager

import torch
from torch.utils.data import DataLoader

from generate_diffusion_building import draw_scene, draw_scene_list
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

    # print(args.experiment_tag)
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
    shutil.copy(args.config_file, os.path.join(experiment_directory, "config.yaml"))
    print("Save experiment statistics in {}".format(experiment_directory))

    # Parse the config file
    config = load_config(args.config_file)

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
        # data_cache=None
    )
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

    use_ddp = torch.cuda.device_count() > 1

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
        ddp=use_ddp,
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
    # import ipdb;ipdb.set_trace()

    load_checkpoints(network, optimizer, experiment_directory, args, device)

    if use_ddp:
        print("Using", torch.cuda.device_count(), "GPUs!")
        network = torch.nn.DataParallel(network)
    network.to(device)

    # Load the learning rate scheduler
    lr_scheduler = schedule_factory(config["training"])

    # import ipdb;ipdb.set_trace()
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
    # StatsLogger.instance().add_output_file(open(
    #     os.path.join(experiment_directory, "stats.txt"),
    #     "w"
    # ))

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
            for k, v in sample.items():
                if not isinstance(v, list):
                    sample[k] = v.to(device)
            # print("aaaaaa")
            # print(len(sample['class_labels']))
            # print(len(sample['description']))
            if use_ddp:
                batch_loss = train_on_batch(network, optimizer, sample, config)
            else:
                batch_loss = train_on_batch(network, optimizer, sample, config)
            batch_time_now = time.time() - batch_start_time  # Calculate batch time
            batch_start_time = time.time()  # Start timing the batch
            if batch_time_ema == 0:
                batch_time_ema = batch_time_now
            batch_time_ema = batch_time_ema * 0.99 + batch_time_now * 0.01
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
        StatsLogger.instance().clear()

        if i % val_every == 0 and i > 0:
            print("====> Validation Epoch ====>")
            network.eval()
            val_batch = 4
            text = [
                " This is an American country-style building. It has a rectangular shape and a brick exterior. It is two stories tall, with a full-width porch on the second story and a half-width porch on the first story. The roof is flat and has a slight overhang. There are five windows and two doors on the front of the building. The windows are all rectangular and have wooden shutters. The doors are both wooden and have a rectangular shape. There is a chimney on the right side of the building. There is a wooden fence around the property.",
                " This is a Simple European-style building. It is a three-story house. The house is made up of a rectangular prism for the main structure with a rectangular prism bump-out on the left side and a triangular roof. There are five rectangular windows and one rectangular door on the front of the house. There is one rectangular window on the left side of the house and one on the right side. There are two rectangular windows on the back of the house. There is a chimney on the back of the roof. There is a fence between the house and the sidewalk.",
                " This is a Simple European-style building. It is a rectangular building with a pitched roof. The roof has two slopes that meet in the center and is covered in red tiles. The front of the building has two stories. The first story has a door in the center with two windows to the left and right. The second story has three windows. The left and right sides of the building each have two windows on each story. The back of the building has a door in the center with a window to the left. The second story has two windows.",
                " This is a Chinese-style building. It has an irregular shape, with the front facing slightly left. The building has two floors. The roof has two sections. The larger section is a hip roof with four slopes. The smaller section is a gable roof with two slopes. Both sections are covered in moss. The walls are made of wood. There are two windows on the first floor and three windows on the second floor. The windows have wooden frames and paper panes. The door is located on the first floor and has a wooden frame and paper panels. There is a chimney on the roof. The building has a balcony on the second floor. The balcony has a wooden railing. There are two pillars supporting the roof. The pillars are made of wood.",
            ] * torch.cuda.device_count()
            bbox_params = network.generate_layout(
                room_mask=torch.zeros([val_batch, 64]).to(device),
                batch_size=val_batch,
                num_points=config["network"]["sample_num_points"],
                point_dim=config["network"]["point_dim"],
                # text=torch.from_numpy(samples['desc_emb'])[None, :].to(device) if 'desc_emb' in samples.keys() else None, # glove embedding
                # text=samples['description'] if 'description' in samples.keys() else None,  # bert
                text=text if "description" in sample.keys() else None,
                device=device,
                clip_denoised=True,
                batch_seeds=torch.arange(0, 1),
            )
            classes = np.array(validation_dataset.class_labels)

            boxes = [
                validation_dataset.post_process(bbox_param)
                for bbox_param in bbox_params
            ]
            draw_scene_list(
                boxes,
                classes,
                save_path=os.path.join(
                    args.output_directory, "valEpoch" + str(i) + ".png"
                ),
                size_half=size_half,
            )

            for b, sample in enumerate(val_loader):
                # Move everything to device
                for k, v in sample.items():
                    if not isinstance(v, list):
                        sample[k] = v.to(device)
                if use_ddp:
                    batch_loss = validate_on_batch(network.module, sample, config)
                else:
                    batch_loss = validate_on_batch(network, sample, config)
                StatsLogger.instance().print_progress(-1, b + 1, batch_loss)
            StatsLogger.instance().clear()
            print("====> Validation Epoch ====>")


if __name__ == "__main__":
    main(sys.argv[1:])
