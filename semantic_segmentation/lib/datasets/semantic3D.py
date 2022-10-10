import os
import logging
import numpy as np

from lib.dataset import VoxelizationDataset, DatasetPhase, str2datasetphase_type
from lib.pc_utils import read_plyfile, save_point_cloud
from lib.utils import read_txt, fast_hist, per_class_iu



train_file_prefixes = [
    # "bildstein_station1_xyz_intensity_rgb",
    # "bildstein_station3_xyz_intensity_rgb",
    # "bildstein_station5_xyz_intensity_rgb",
    # "domfountain_station1_xyz_intensity_rgb",
    # "domfountain_station2_xyz_intensity_rgb",
    # "domfountain_station3_xyz_intensity_rgb",
    # "neugasse_station1_xyz_intensity_rgb",
    # "sg27_station1_intensity_rgb",
    # "sg27_station2_intensity_rgb",
]

validation_file_prefixes = [
    # "sg27_station4_intensity_rgb",
    "sg27_station5_intensity_rgb",
    "sg27_station9_intensity_rgb",
    "sg28_station4_intensity_rgb",
    "untermaederbrunnen_station1_xyz_intensity_rgb",
    "untermaederbrunnen_station3_xyz_intensity_rgb",
]

test_file_prefixes = [
    # "birdfountain_station1_xyz_intensity_rgb",
    # "castleblatten_station1_intensity_rgb",
    # "castleblatten_station5_xyz_intensity_rgb",
    # "marketplacefeldkirch_station1_intensity_rgb",
    # "marketplacefeldkirch_station4_intensity_rgb",
    # "marketplacefeldkirch_station7_intensity_rgb",
    # "sg27_station10_intensity_rgb",
    # "sg27_station3_intensity_rgb",
    # "sg27_station6_intensity_rgb",
    # "sg27_station8_intensity_rgb",
    # "sg28_station2_intensity_rgb",
    # "sg28_station5_xyz_intensity_rgb",
    # "stgallencathedral_station1_intensity_rgb",
    # "stgallencathedral_station3_intensity_rgb",
    # "stgallencathedral_station6_intensity_rgb",
]

all_file_prefixes = train_file_prefixes + validation_file_prefixes + test_file_prefixes

CLASS_LABELS = ("unlabeled",
            "man-made terrain",
            "natural terrain",
            "high vegetation",
            "low vegetation",
            "buildings",
            "hard scape",
            "scanning artifact",
            "cars")
VALID_CLASS_IDS = (1,2,3,4,5,6,7,8)


class SemanticVoxelizationDataset(VoxelizationDataset):

    # Voxelization arguments
    CLIP_BOUND = None
    TEST_CLIP_BOUND = None
    VOXEL_SIZE = 0.5

    # Augmentation arguments
    ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi,
                                                                                            np.pi))
    TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
    ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))

    ROTATION_AXIS = 'z'
    LOCFEAT_IDX = 2
    NUM_LABELS = 9  # Will be converted to 20 as defined in IGNORE_LABELS.
    IGNORE_LABELS = tuple(set(range(NUM_LABELS)) - set(VALID_CLASS_IDS))
    IS_FULL_POINTCLOUD_EVAL = True

    DATA_PATH_FILE = {
        DatasetPhase.Train: train_file_prefixes,
        DatasetPhase.Val: validation_file_prefixes,
        DatasetPhase.TrainVal: train_file_prefixes + validation_file_prefixes,
        DatasetPhase.Test: test_file_prefixes
    }


    # def __init__(
    #     self, 
    #     num_points_per_sample, 
    #     split, 
    #     use_color, 
    #     box_size_x, 
    #     box_size_y, 
    #     path
    # ):
    def __init__(self,
        config,
        prevoxel_transform=None,
        input_transform=None,
        target_transform=None,
        augment_data=True,
        elastic_distortion=False,
        cache=False,
        phase=DatasetPhase.Train):
        """Create a dataset holder
        num_points_per_sample (int): Defaults to 8192. The number of point per sample
        split (str): Defaults to 'train'. The selected part of the data (train, test,
                     reduced...)
        color (bool): Defaults to True. Whether to use colors or not
        box_size_x (int): Defaults to 10. The size of the extracted cube.
        box_size_y (int): Defaults to 10. The size of the extracted cube.
        path (float): Defaults to 'dataset/semantic_data/'.
        """
        if isinstance(phase, str):
            phase = str2datasetphase_type(phase)
        data_root = config.semantic3d_path
        if phase not in [DatasetPhase.Train, DatasetPhase.TrainVal]:
            self.CLIP_BOUND = self.TEST_CLIP_BOUND
        data_paths = read_txt(os.path.join('./splits/semantic3d', self.DATA_PATH_FILE[phase]))
        logging.info('Loading {} {}: {}'.format(self.__class__.__name__, phase,
                                            self.DATA_PATH_FILE[phase]))
        super().__init__(
            data_paths,
            data_root=data_root,
            prevoxel_transform=prevoxel_transform,
            input_transform=input_transform,
            target_transform=target_transform,
            ignore_label=config.ignore_label,
            return_transformation=config.return_transformation,
            augment_data=augment_data,
            elastic_distortion=elastic_distortion,
            config=config)
        
        # Dataset parameters
    #     self.num_points_per_sample = num_points_per_sample
    #     self.split = split
    #     self.use_color = use_color
    #     self.box_size_x = box_size_x
    #     self.box_size_y = box_size_y
    #     self.num_classes = 9
    #     self.path = path
    #     self.labels_names = [
    #         "unlabeled",
    #         "man-made terrain",
    #         "natural terrain",
    #         "high vegetation",
    #         "low vegetation",
    #         "buildings",
    #         "hard scape",
    #         "scanning artifact",
    #         "cars",
    #     ]

    #     # Get file_prefixes
    #     file_prefixes = map_name_to_file_prefixes[self.split]
    #     print("Dataset split:", self.split)
    #     print("Loading file_prefixes:", file_prefixes)

    #     # Load files
    #     self.list_file_data = []
    #     for file_prefix in file_prefixes:
    #         file_path_without_ext = os.path.join(self.path, file_prefix)
    #         file_data = SemanticFileData(
    #             file_path_without_ext=file_path_without_ext,
    #             has_label=self.split != "test",
    #             use_color=self.use_color,
    #             box_size_x=self.box_size_x,
    #             box_size_y=self.box_size_y,
    #         )
    #         self.list_file_data.append(file_data)

    #     # Pre-compute the probability of picking a scene
    #     self.num_scenes = len(self.list_file_data)
    #     self.scene_probas = [
    #         len(fd.points) / self.get_total_num_points() for fd in self.list_file_data
    #     ]

    #     # Pre-compute the points weights if it is a training set
    #     if self.split == "train" or self.split == "train_full":
    #         # First, compute the histogram of each labels
    #         label_weights = np.zeros(9)
    #         for labels in [fd.labels for fd in self.list_file_data]:
    #             tmp, _ = np.histogram(labels, range(10))
    #             label_weights += tmp

    #         # Then, a heuristic gives the weights
    #         # 1 / log(1.2 + probability of occurrence)
    #         label_weights = label_weights.astype(np.float32)
    #         label_weights = label_weights / np.sum(label_weights)
    #         self.label_weights = 1 / np.log(1.2 + label_weights)
    #     else:
    #         self.label_weights = np.zeros(9)

    # def sample_batch_in_all_files(self, batch_size, augment=True):
    #     batch_data = []
    #     batch_label = []
    #     batch_weights = []

    #     for _ in range(batch_size):
    #         points, labels, colors, weights = self.sample_in_all_files(is_training=True)
    #         if self.use_color:
    #             batch_data.append(np.hstack((points, colors)))
    #         else:
    #             batch_data.append(points)
    #         batch_label.append(labels)
    #         batch_weights.append(weights)

    #     batch_data = np.array(batch_data)
    #     batch_label = np.array(batch_label)
    #     batch_weights = np.array(batch_weights)

    #     if augment:
    #         if self.use_color:
    #             batch_data = provider.rotate_feature_point_cloud(batch_data, 3)
    #         else:
    #             batch_data = provider.rotate_point_cloud(batch_data)

    #     return batch_data, batch_label, batch_weights

    # def sample_in_all_files(self, is_training):
    #     """
    #     Returns points and other info within a z - cropped box.
    #     """
    #     # Pick a scene, scenes with more points are more likely to be chosen
    #     scene_index = np.random.choice(
    #         np.arange(0, len(self.list_file_data)), p=self.scene_probas
    #     )

    #     # Sample from the selected scene
    #     points_centered, points_raw, labels, colors = self.list_file_data[
    #         scene_index
    #     ].sample(num_points_per_sample=self.num_points_per_sample)

    #     if is_training:
    #         weights = self.label_weights[labels]
    #         return points_centered, labels, colors, weights
    #     else:
    #         return scene_index, points_centered, points_raw, labels, colors

    # def get_total_num_points(self):
    #     list_num_points = [len(fd.points) for fd in self.list_file_data]
    #     return np.sum(list_num_points)

    # def get_num_batches(self, batch_size):
    #     return int(
    #         self.get_total_num_points() / (batch_size * self.num_points_per_sample)
    #     )

    # def get_file_paths_without_ext(self):
    #     return [file_data.file_path_without_ext for file_data in self.list_file_data]
