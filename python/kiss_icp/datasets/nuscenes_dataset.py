# MIT License
#
# Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
# Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import importlib
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
from pyquaternion import Quaternion

class NuScenesDataset:
    def __init__(self, data_dir: Path, sequence: str, remove_dynamic: bool = False, *_, **__):
        try:
            importlib.import_module("nuscenes")
        except ModuleNotFoundError:
            print("nuscenes-devkit is not installed on your system")
            print('run "pip install nuscenes-devkit"')
            sys.exit(1)

        # TODO: If someone needs more splits from nuScenes expose this 2 parameters
        nusc_version: str = "v1.0-trainval"
        # nusc_version: str = "v1.0-mini"
        self.lidar_name: str = "LIDAR_TOP"
        self.remove_dynamic = remove_dynamic

        # Lazy loading
        from nuscenes.nuscenes import NuScenes

        self.sequence_id = str(sequence).zfill(4)

        self.nusc = NuScenes(dataroot=str(data_dir), version=nusc_version)
        scene_name = f"scene-{self.sequence_id}"
        if scene_name not in [s["name"] for s in self.nusc.scene]:
            print(f'[ERROR] Sequence "{self.sequence_id}" not available scenes')
            print("\nAvailable scenes:")
            self.nusc.list_scenes()
            sys.exit(1)
        scene_token = [s["token"] for s in self.nusc.scene if s["name"] == scene_name][0]

        # Load nuScenes read from file inside dataloader module
        self.load_point_cloud = importlib.import_module(
            "nuscenes.utils.data_classes"
        ).LidarPointCloud.from_file

        # Use only the samples from the current split.
        self.lidar_tokens, self.timestamps = self._get_lidar_tokens(scene_token)
        self.gt_poses = self._load_poses()

    def __len__(self):
        return len(self.lidar_tokens)

    def __getitem__(self, idx):
        return self.read_point_cloud(self.lidar_tokens[idx])

    def read_point_cloud(self, token: str):
        filename = self.nusc.get("sample_data", token)["filename"]
        pcl = self.load_point_cloud(os.path.join(self.nusc.dataroot, filename))
        points = pcl.points.T[:, :3]

        # Filter out points belonging to dynamic objects
        if self.remove_dynamic:
            points = self.remove_dynamic_objects(points, token)

        return points.astype(np.float64)

    def remove_dynamic_objects(self, points: np.ndarray, lidar_token: str) -> np.ndarray:
        sample_data = self.nusc.get('sample_data', lidar_token)

        # Get sensor pose
        sensor_pose = self.nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
        sensor_rotation = Quaternion(sensor_pose['rotation'])
        sensor_translation = np.array(sensor_pose['translation'])

        # Get ego pose
        ego_pose = self.nusc.get('ego_pose', sample_data['ego_pose_token'])
        ego_rotation = Quaternion(ego_pose['rotation'])
        ego_translation = np.array(ego_pose['translation'])

        from nuscenes.utils.geometry_utils import transform_matrix

        # Combined transformation: sensor -> ego -> global
        sensor_to_global = transform_matrix(ego_translation, ego_rotation).dot(
            transform_matrix(sensor_translation, sensor_rotation)
        )

        # Transform points to global coordinate system
        points_homogeneous = np.hstack((points, np.ones((len(points), 1))))
        points_global = (sensor_to_global @ points_homogeneous.T).T[:, :3]

        # Get all annotations for this sample
        boxes = self.nusc.get_boxes(lidar_token)

        # Remove points inside dynamic object boxes
        mask = np.ones(len(points), dtype=bool)
        margin = 1e-2
        for box in boxes:
            box_points = points_global - box.center
            box_points = np.dot(Quaternion(matrix=box.rotation_matrix).inverse.rotation_matrix, box_points.T).T

            half_size = box.wlh[[1,0,2]] / 2
            mask = np.logical_and(mask, np.any(np.abs(box_points) > (half_size + margin), axis=1))

        return points[mask]

    def _load_poses(self) -> np.ndarray:
        from nuscenes.utils.geometry_utils import transform_matrix
        from pyquaternion import Quaternion

        poses = np.empty((len(self), 4, 4), dtype=np.float32)
        for i, lidar_token in enumerate(self.lidar_tokens):
            sd_record_lid = self.nusc.get("sample_data", lidar_token)
            cs_record_lid = self.nusc.get(
                "calibrated_sensor", sd_record_lid["calibrated_sensor_token"]
            )
            ep_record_lid = self.nusc.get("ego_pose", sd_record_lid["ego_pose_token"])

            self.car_to_velo = transform_matrix(
                cs_record_lid["translation"],
                Quaternion(cs_record_lid["rotation"]),
            )
            pose_car = transform_matrix(
                ep_record_lid["translation"],
                Quaternion(ep_record_lid["rotation"]),
            )

            poses[i:, :] = pose_car @ self.car_to_velo

        # Convert from global coordinate poses to local poses
        self.first_pose = poses[0, :, :]
        poses = np.linalg.inv(self.first_pose) @ poses
        return poses

    def _get_lidar_tokens(self, scene_token: str) -> List[str]:
        # Get records from DB.
        scene_rec = self.nusc.get("scene", scene_token)
        start_sample_rec = self.nusc.get("sample", scene_rec["first_sample_token"])
        sd_rec = self.nusc.get("sample_data", start_sample_rec["data"][self.lidar_name])

        # Make list of frames
        cur_sd_rec = sd_rec
        sd_tokens = [cur_sd_rec["token"]]
        timestamps = [cur_sd_rec["timestamp"]]
        while cur_sd_rec["next"] != "":
            cur_sd_rec = self.nusc.get("sample_data", cur_sd_rec["next"])
            sd_tokens.append(cur_sd_rec["token"])
            timestamps.append(cur_sd_rec["timestamp"])
        return sd_tokens, timestamps

    def apply_calibration(self, poses):
        # lid_to_first_lid = poses
        # first_lid_to_global = self.first_pose
        # lid_to_global = first_lid_to_global @ lid_to_first_lid
        # lid_to_ego = self.car_to_velo
        # ego_to_lid = np.linalg.inv(lid_to_ego)
        # ego_to_global = ego_to_lid @ lid_to_global
        # return ego_to_global
        # convert from local lidar coordinate to global ego coordinate
        return self.first_pose @ poses @ np.linalg.inv(self.car_to_velo)

    def get_frames_timestamps(self):
        return self.timestamps
