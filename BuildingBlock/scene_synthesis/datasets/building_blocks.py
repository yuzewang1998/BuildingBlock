from collections import Counter, OrderedDict
from functools import lru_cache
from tkinter import E
import numpy as np
import json
import os

from PIL import Image

from .common import BaseDataset
from .threed_front_scene import Room
from .utils import parse_threed_front_scenes


class BuildingBlocks(BaseDataset):
    """Container for the scenes in the 3D-FRONT dataset.

    Arguments
    ---------
    scenes: list of Room objects for all scenes in 3D-FRONT dataset
    """

    def __init__(self, scenes, bounds=None):
        super().__init__(scenes)
        assert isinstance(self.scenes[0], Room)
        self._object_types = None
        self._room_types = None
        self._count_furniture = None
        self._bbox = None

        self._sizes = self._centroids = self._angles = self._objfeats = None
        if bounds is not None:
            self._sizes = bounds["sizes"]
            self._centroids = bounds["translations"]
            self._angles = bounds["angles"]
            if "objfeats" in bounds.keys():
                self._objfeats = bounds["objfeats"]
            else:
                self._objfeats = (
                    np.array([1]),
                    np.array([-1]),
                    np.array([1]),
                )  # std, min, max

            if "objfeats_32" in bounds.keys():
                self._objfeats_32 = bounds["objfeats_32"]
            else:
                self._objfeats_32 = (
                    np.array([1]),
                    np.array([-1]),
                    np.array([1]),
                )  # std, min, max

        self._max_length = None

    def __str__(self):
        return "Dataset contains {} scenes with {} discrete types".format(
            len(self.scenes), self.n_object_types
        )

    @property
    def bbox(self):
        """The bbox for the entire dataset is simply computed based on the
        bounding boxes of all scenes in the dataset.
        """
        if self._bbox is None:
            _bbox_min = np.array([1000, 1000, 1000])
            _bbox_max = np.array([-1000, -1000, -1000])
            for s in self.scenes:
                bbox_min, bbox_max = s.bbox
                _bbox_min = np.minimum(bbox_min, _bbox_min)
                _bbox_max = np.maximum(bbox_max, _bbox_max)
            self._bbox = (_bbox_min, _bbox_max)
        return self._bbox

    def _centroid(self, box, offset):
        return box.centroid(offset)

    def _size(self, box):
        return box.size

    def _compute_bounds(self):
        _size_min = np.array([10000000] * 3)
        _size_max = np.array([-10000000] * 3)
        _centroid_min = np.array([10000000] * 3)
        _centroid_max = np.array([-10000000] * 3)
        _angle_min = np.array([10000000000])
        _angle_max = np.array([-10000000000])
        _objfeat_std = np.array([1])
        _objfeat_min = np.array([10000000000])
        _objfeat_max = np.array([-10000000000])
        _objfeat_32_std = np.array([1])
        _objfeat_32_min = np.array([10000000000])
        _objfeat_32_max = np.array([-10000000000])
        all_objfeats = []
        all_objfeats_32 = []
        for s in self.scenes:
            for f in s.bboxes:
                if np.any(f.size > 5):
                    print(s.scene_id, f.size, f.model_uid, f.scale)
                centroid = self._centroid(f, -s.centroid)
                _centroid_min = np.minimum(centroid, _centroid_min)
                _centroid_max = np.maximum(centroid, _centroid_max)
                _size_min = np.minimum(self._size(f), _size_min)
                _size_max = np.maximum(self._size(f), _size_max)
                _angle_min = np.minimum(f.z_angle, _angle_min)
                _angle_max = np.maximum(f.z_angle, _angle_max)
                all_objfeats.append(f.raw_model_norm_pc_lat())
                all_objfeats_32.append(f.raw_model_norm_pc_lat32())

        all_objfeats = np.stack(all_objfeats, axis=0)
        all_objfeats_32 = np.stack(all_objfeats_32, axis=0)
        _objfeat_std, _objfeat_min, _objfeat_max = (
            np.array([all_objfeats.flatten().std()]),
            np.array([all_objfeats.min()]),
            np.array([all_objfeats.max()]),
        )
        _objfeat_32_std, _objfeat_32_min, _objfeat_32_max = (
            np.array([all_objfeats_32.flatten().std()]),
            np.array([all_objfeats_32.min()]),
            np.array([all_objfeats_32.max()]),
        )

        self._sizes = (_size_min, _size_max)
        self._centroids = (_centroid_min, _centroid_max)
        self._angles = (_angle_min, _angle_max)
        self._objfeats = (_objfeat_std, _objfeat_min, _objfeat_max)
        self._objfeats_32 = (_objfeat_32_std, _objfeat_32_min, _objfeat_32_max)

    @property
    def bounds(self):
        return {
            "translations": self.centroids,
            "sizes": self.sizes,
            "angles": self.angles,
            "objfeats": self.objfeats,
            "objfeats_32": self.objfeats_32,
        }

    @property
    def sizes(self):
        if self._sizes is None:
            self._compute_bounds()
        return self._sizes

    @property
    def centroids(self):
        if self._centroids is None:
            self._compute_bounds()
        return self._centroids

    @property
    def angles(self):
        if self._angles is None:
            self._compute_bounds()
        return self._angles

    @property
    def objfeats(self):
        if self._objfeats is None:
            self._compute_bounds()
        return self._objfeats

    @property
    def objfeats_32(self):
        if self._objfeats_32 is None:
            self._compute_bounds()
        return self._objfeats_32

    @property
    def count_furniture(self):
        if self._count_furniture is None:
            counts = []
            for s in self.scenes:
                counts.append(s.furniture_in_room)
            counts = Counter(sum(counts, []))
            counts = OrderedDict(sorted(counts.items(), key=lambda x: -x[1]))
            self._count_furniture = counts
        return self._count_furniture

    @property
    def class_order(self):
        return dict(zip(self.count_furniture.keys(), range(len(self.count_furniture))))

    @property
    def class_frequencies(self):
        object_counts = self.count_furniture
        class_freq = {}
        n_objects_in_dataset = sum([object_counts[k] for k, v in object_counts.items()])
        for k, v in object_counts.items():
            class_freq[k] = object_counts[k] / n_objects_in_dataset
        return class_freq

    @property
    def object_types(self):
        if self._object_types is None:
            self._object_types = set()
            for s in self.scenes:
                self._object_types |= set(s.object_types)
            self._object_types = sorted(self._object_types)
        return self._object_types

    @property
    def room_types(self):
        if self._room_types is None:
            self._room_types = set([s.scene_type for s in self.scenes])
        return self._room_types

    @property
    def class_labels(self):
        return self.object_types + ["start", "end"]

    # compute max_lenght for diffusion models
    @property
    def max_length(self):
        if self._max_length is None:
            _room_types = set([str(s.scene_type) for s in self.scenes])
            if "bed" in _room_types:
                self._max_length = 12
            elif "living" in _room_types:
                self._max_length = 21
            elif "dining" in _room_types:
                self._max_length = 21
            elif "library" in _room_types:
                self._max_length = 11

        return self._max_length

    @classmethod
    def from_dataset_directory(
        cls,
        dataset_directory,
        path_to_model_info,
        path_to_models,
        path_to_room_masks_dir=None,
        path_to_bounds=None,
        filter_fn=lambda s: s,
    ):
        scenes = parse_threed_front_scenes(
            dataset_directory,
            path_to_model_info,
            path_to_models,
            path_to_room_masks_dir,
        )
        bounds = None
        if path_to_bounds:
            bounds = np.load(path_to_bounds, allow_pickle=True)

        return cls([s for s in map(filter_fn, scenes) if s], bounds)


class CachedBuilding(object):
    def __init__(
        self,
        scene_id,
        class_labels,
        translations,
        sizes,
        angles,
        objfeats,
        objfeats_32,
        image_path,
        condition_floder,
    ):
        self.scene_id = scene_id
        self.class_labels = class_labels
        self.translations = translations
        self.sizes = sizes
        self.angles = angles
        self.objfeats = objfeats
        self.objfeats_32 = objfeats_32
        self.image_path = image_path
        self.condition_floder = condition_floder


class CachedBuildingBlocks(BuildingBlocks):
    def __init__(self, base_dir, config, scene_ids):
        self._base_dir = base_dir
        self.config = config
        # add it for scene_type
        self.cond_path = config.get("cond_path", "renderFirstBatch")
        self._parse_train_stats(config["train_stats"])
        self._tags = sorted(
            [oi for oi in os.listdir(self._base_dir) if oi.split("_A")[0] in scene_ids]
        )
        self._path_to_rooms = sorted(
            [os.path.join(self._base_dir, pi, "boxes.npz") for pi in self._tags]
        )

        rendered_scene = "rendered_scene_256.png"
        path_to_rendered_scene = os.path.join(
            self._base_dir, self._tags[0], rendered_scene
        )
        if not os.path.isfile(path_to_rendered_scene):
            rendered_scene = "rendered_scene_256_no_lamps.png"

        path_to_rendered_scene = os.path.join(
            self._base_dir, self._tags[0], rendered_scene
        )
        if not os.path.isfile(path_to_rendered_scene):
            rendered_scene = "rendered_scene_notexture_256.png"
        print("rendered_scene is :", rendered_scene)
        self._path_to_renders = sorted(
            [os.path.join(self._base_dir, pi, rendered_scene) for pi in self._tags]
        )
        self._path_of_condition = sorted(
            [
                os.path.join(self._base_dir, self.cond_path, pi.split("_A")[0])
                for pi in self._tags
            ]
        )

    def _get_room_layout(self, room_layout):
        img = Image.fromarray(room_layout[:, :, 0])
        img = img.resize(
            tuple(map(int, self.config["room_layout_size"].split(","))),
            resample=Image.BILINEAR,
        )
        D = np.asarray(img).astype(np.float32) / np.float32(255)
        return D

    def _get_room_rgb_2d(self, img_path):
        img = Image.open(img_path)
        img = img.resize(
            tuple(map(int, self.config["room_layout_size"].split(","))),
            resample=Image.BILINEAR,
        )
        D = np.asarray(img).astype(np.float32) / np.float32(255)
        return D

    # @lru_cache(maxsize=32)
    def __getitem__(self, i):
        D = np.load(self._path_to_rooms[i])
        return CachedBuilding(
            scene_id=D["scene_id"],
            class_labels=D["class_labels"],
            translations=D["translations"],
            sizes=D["sizes"],
            angles=D["angles"],
            objfeats=None,
            objfeats_32=None,
            image_path=self._path_to_renders[i],
            condition_floder=self._path_of_condition[i],
        )

    def get_room_params(self, i):
        D = np.load(self._path_to_rooms[i])
        img_condition = self.config.get("img_condition", False)
        txt_condition = self.config.get("txt_condition", False)
        cut_text = self.config.get("cut_text", False)

        txt_suffix = self.config.get("txt_suffix", ".txt")

        # import ipdb;ipdb.set_trace()
        texts = []
        if txt_condition:
            cond_list = os.listdir(self._path_of_condition[i])
            cond_list = sorted(cond_list)
            txt_list = [cond for cond in cond_list if cond.endswith(txt_suffix)]

            for txt in txt_list:
                describe = self._read_condition_text(
                    os.path.join(self._path_of_condition[i], txt)
                )
                if not describe:
                    continue
                if cut_text:
                    describe = describe.split(".")
                    sentences = np.random.randint(
                        1, 4 if len(describe) > 4 else len(describe)
                    )
                    describe = describe[:sentences]
                    describe = ".".join(describe)
                describe = describe.strip()
                if describe:
                    texts.append(describe)

        data_dict = {
            # "room_layout": room,
            "class_labels": D["class_labels"],
            "translations": D["translations"],
            "sizes": D["sizes"],
            "angles": D["angles"],
            "texts": texts,
        }

        # if "objfeats" in D.keys():
        #     data_dict[ "objfeats" ] = D["objfeats"]
        # if "objfeats_32" in D.keys():
        #     data_dict[ "objfeats_32" ] = D["objfeats_32"]

        return data_dict

    @staticmethod
    def _read_condition_text(path):
        for encoding in ("utf-8", "utf-8-sig", "latin1"):
            try:
                with open(path, "r", encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
            except OSError:
                return None
        return None

    def __len__(self):
        return len(self._path_to_rooms)

    def __str__(self):
        return "Dataset contains {} scenes with {} discrete types".format(
            len(self), self.n_object_types
        )

    def _parse_train_stats(self, train_stats):
        with open(os.path.join(self._base_dir, train_stats), "r") as f:
            train_stats = json.load(f)

        self._centroids = train_stats["bounds_translations"]
        self._centroids = (np.array(self._centroids[:3]), np.array(self._centroids[3:]))
        self._sizes = train_stats["bounds_sizes"]
        self._sizes = (np.array(self._sizes[:3]), np.array(self._sizes[3:]))
        self._angles = train_stats["bounds_angles"]
        self._angles = (np.array(self._angles[0]), np.array(self._angles[1]))

        if "bounds_objfeats" in train_stats.keys():
            self._objfeats = train_stats["bounds_objfeats"]
            print("bounds_objfeats of dataset:", self._objfeats)
            self._objfeats = (
                np.array([self._objfeats[0]]),
                np.array([self._objfeats[1]]),
                np.array([self._objfeats[2]]),
            )
        else:
            self._objfeats = (np.array([1]), np.array([-1]), np.array([1]))

        if "bounds_objfeats_32" in train_stats.keys():
            self._objfeats_32 = train_stats["bounds_objfeats_32"]
            print("bounds_objfeats_32 of dataset:", self._objfeats_32)
            self._objfeats_32 = (
                np.array([self._objfeats_32[0]]),
                np.array([self._objfeats_32[1]]),
                np.array([self._objfeats_32[2]]),
            )
        else:
            self._objfeats_32 = (np.array([1]), np.array([-1]), np.array([1]))

        self._class_labels = train_stats["class_labels"]
        self._object_types = train_stats["object_types"]
        self._class_frequencies = train_stats["class_frequencies"]
        self._class_order = train_stats["class_order"]
        self._count_furniture = train_stats["count_furniture"]
        self._max_length = self.config.get("max_length", 12)
        self._pad_zero = self.config.get("pad_zero", True)

    @property
    def class_labels(self):
        return self._class_labels

    @property
    def object_types(self):
        return self._object_types

    @property
    def class_frequencies(self):
        return self._class_frequencies

    @property
    def class_order(self):
        return self._class_order

    @property
    def count_furniture(self):
        return self._count_furniture

    # compute max_lenght for diffusion models
    @property
    def max_length(self):
        return self._max_length

    @property
    def pad_zero(self):
        return self._pad_zero
