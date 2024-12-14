import os
from math import ceil, floor
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tensorflow.keras.utils import Sequence

# (class, x_center, y_center, width, height)
BoundingBoxYOLO = Tuple[float, float, float, float, float]
# (class, x_min, y_min, x_max, y_max)
BoundingBoxPascalVOC = Tuple[float, int, int, int, int]


# (class, x_center, y_center, width, height)
BoundingBoxYOLO = Tuple[float, float, float, float, float]
# (class, x_min, y_min, x_max, y_max)
BoundingBoxPascalVOC = Tuple[float, int, int, int, int]


class YOLOv8DataGenerator(Sequence):
    """
    Data generator for YOLOv8 datasets using the Keras Sequence class.

    Args:
        images_dir (str): Path to the images directory.
        labels_dir (str): Path to the labels directory.
        batch_size (int): Number of samples per batch.
        target_size (Tuple[int, int]): Target size for resizing images (height, width).
        shuffle (bool): Whether to shuffle the data at the start of each epoch.
    """

    def __init__(self, images_dir: str, labels_dir: str, batch_size: int = 32, target_size: Tuple[int, int] = (640, 640), shuffle: bool = True,):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle

        # Filter for valid image files
        valid_extensions = (".jpg", ".jpeg", ".png")
        self.image_files = [
            f for f in os.listdir(images_dir)
            if os.path.isfile(os.path.join(images_dir, f)) and f.lower().endswith(valid_extensions)
        ]
        if not self.image_files:
            raise ValueError(
                f"No valid image files found in directory: {images_dir}")

        # Print dataset and batch information
        total_images = len(self.image_files)
        total_batches = int(np.ceil(total_images / self.batch_size))
        print(f"Total images in dataset: {total_images}")
        print(f"Batch size: {self.batch_size}")
        print(f"Total batches per epoch: {total_batches}")

        # Warn if batch size is larger than dataset size
        if self.batch_size > total_images:
            print(f"Warning: Batch size ({
                self.batch_size}) is larger than the dataset size ({total_images}).")

        self.on_epoch_end()

    def __len__(self) -> int:
        """
        Returns the number of batches per epoch, accounting for incomplete final batches.
        """
        return ceil(len(self.image_files) / self.batch_size)  # Use ceil to include partial final batch

    def __getitem__(self, index: int) -> Tuple[np.ndarray, List[List[BoundingBoxYOLO]]]:
        """
        Generates one batch of data.

        Args:
            index (int): Batch index.

        Returns:
            Tuple[np.ndarray, List[List[BoundingBoxYOLO]]]: A batch of images and labels.
        """
        if index < 0 or index >= len(self):
            raise IndexError(
                f"Index {index} is out of bounds for batch size {len(self)}.")

        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, len(self.image_files))
        batch_files = self.image_files[start:end]

        images, labels = self.__load_batch(batch_files)

        if len(images) == 0:
            raise ValueError(
                f"Batch {index} is empty. Ensure your dataset contains valid images.")

        return images, labels

    def on_epoch_end(self):
        """
        Shuffles the data at the end of each epoch, if shuffle is enabled.
        """
        if self.shuffle:
            np.random.shuffle(self.image_files)

    def __load_batch(self, batch_files):
        """
        Loads and preprocesses a batch of images.

        Args:
            batch_files (List[str]): List of filenames in the current batch.

        Returns:
            Tuple[np.ndarray, List]: Array of preprocessed images and empty labels for autoencoder training.
        """
        images = []

        for img_file in batch_files:
            img_path = os.path.join(self.images_dir, img_file)

            # Load and preprocess image
            img = cv2.imread(img_path)
            if img is None:
                print(
                    f"Warning: Skipping invalid or corrupted image: {img_file}")
                continue

            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize image
            try:
                img_resized = cv2.resize(img, self.target_size)
            except Exception as e:
                print(f"Warning: Error resizing image {img_file}: {e}")
                continue

            images.append(img_resized.astype(np.float32) /
                          255.0)  # Normalize to [0, 1]

        if not images:
            print(f"Warning: Batch is empty for files: {batch_files}")

        return np.array(images), []


def yolo_to_pascal_voc(bboxes: List[BoundingBoxYOLO], img_width: int, img_height: int) -> List[BoundingBoxPascalVOC]:
    """
    Converts bounding boxes from YOLO format to Pascal VOC.

    Args:
        bboxes (List[BoundingBoxYOLO]): List of bounding boxes in YOLO format (class, x_center, y_center, width, height).
        img_width (int): Width of the image.
        img_height (int): Height of the image.

    Returns:
        List[BoundingBoxPascalVOC]: List of bounding boxes in Pascal VOC format (class, x_min, y_min, x_max, y_max).
    """
    pascal_bboxes: List[BoundingBoxPascalVOC] = []
    for bbox in bboxes:
        cls, x_center, y_center, width, height = bbox
        x_min = int((x_center - width / 2) * img_width)
        y_min = int((y_center - height / 2) * img_height)
        x_max = int((x_center + width / 2) * img_width)
        y_max = int((y_center + height / 2) * img_height)
        pascal_bboxes.append((cls, x_min, y_min, x_max, y_max))
    return pascal_bboxes
