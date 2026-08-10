#!/usr/bin/env python3
"""
LeRobot Dataset Converter
Converts HDF5 episode files to LeRobot dataset format for Hugging Face Hub.

Usage:
    python convert_lerobot.py --config config.yaml
"""

import h5py
import numpy as np
import cv2
import yaml
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm
import shutil
import os
import argparse
import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LeRobotDatasetConverter:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.repo_id = config["repo_id"]
        self.fps = config.get("fps", 30)
        self.raw_data_dir = Path(config["raw_data_dir"])
        self.output_dir = Path(config.get("output_dir", os.path.expanduser("~/.cache/huggingface/lerobot"))) / self.repo_id
        self.batch_size = config.get("batch_size", 1)
        
        self.features = {
            "observation.images.camera_high": {
                "dtype": "video",
                "shape": (1024, 1024, 3),
                "names": ["height", "width", "rgb"],
            },
            "observation.images.camera_wrist_right": {
                "dtype": "video",
                "shape": (1024, 1024, 3),
                "names": ["height", "width", "rgb"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (7,),
                "names": {"motors": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]},
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": {"motors": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]},
            },
        }

    def _validate_batch_size(self, num_episodes: int):
        """Check if episode count is divisible by batch size"""
        if num_episodes % self.batch_size != 0:
            raise ValueError(
                f"Cannot divide {num_episodes} episodes into batches of {self.batch_size}. "
                f"Please adjust batch_size in config or number of episodes."
            )
        logger.info(f"Using batch size: {self.batch_size}")

    def create_dataset(self) -> LeRobotDataset:
        """Create dataset with configured batch size"""
        if self.output_dir.exists():
            logger.warning(f"Clearing existing output directory...")
            shutil.rmtree(self.output_dir)
            
        return LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=self.fps,
            features=self.features,
            use_videos=True,
            image_writer_processes=4,
            image_writer_threads=16,
            batch_encoding_size=self.batch_size,  # The key change
        )

    def process_episode(self, hdf5_path: Path, dataset: LeRobotDataset) -> bool:
        """Process single episode (unchanged from original)"""
        try:
            with h5py.File(hdf5_path, "r") as hdf5_file:
                task = hdf5_file.attrs["task"]
                qpos = hdf5_file["observations/qpos"][:]

                actions = np.empty_like(qpos)
                actions[:-1] = qpos[1:]
                actions[-1] = qpos[-1]

                image_data = {
                    cam_name: hdf5_file[f"observations/images/{cam_name}"][:]
                    for cam_name in ["camera_high", "camera_wrist_right"]
                }

                decoded_images = {}
                for cam_name, img_data in image_data.items():
                    decoded_images[cam_name] = [
                        cv2.cvtColor(
                            cv2.imdecode(np.frombuffer(img_bytes, np.uint8), 
                            cv2.IMREAD_COLOR),
                            cv2.COLOR_BGR2RGB
                        )
                        for img_bytes in img_data
                    ]

                min_length = min([len(qpos), len(actions)] + [len(imgs) for imgs in decoded_images.values()])

                for i in range(min_length):
                    dataset.add_frame(
                        {
                            **{f"observation.images.{cam}": decoded_images[cam][i] for cam in decoded_images},
                            "observation.state": qpos[i],
                            "action": actions[i],
                        },
                        task=task
                    )

                dataset.save_episode()
                return True

        except Exception as e:
            logger.error(f"Error processing {hdf5_path.name}: {str(e)}")
            return False

    def run_conversion(self):
        """Run conversion with batch size validation"""
        episode_files = sorted(
            self.raw_data_dir.glob("episode_*.hdf5"),
            key=lambda x: int(x.stem.split("_")[1])
        )
        
        if not episode_files:
            raise FileNotFoundError(f"No episode files found in {self.raw_data_dir}")

        self._validate_batch_size(len(episode_files))

        dataset = self.create_dataset()
        success_count = 0

        for hdf5_file in tqdm(episode_files, desc="Processing episodes"):
            if self.process_episode(hdf5_file, dataset):
                success_count += 1

        logger.info(f"Successfully processed {success_count}/{len(episode_files)} episodes")

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        converter = LeRobotDatasetConverter(config)
        converter.run_conversion()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    main()