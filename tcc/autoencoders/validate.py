import os
from glob import glob

import cv2

cwd = os.getcwd()
data_path = os.path.join(cwd)  # , '..', '..', 'ships_v10i')
img_size = (640, 640)

# Create directories for train data
train_images_dir = os.path.join(data_path, "train/images")
train_labels_dir = os.path.join(data_path, "train/labels")


def validate_dataset(images_dir):
    all_files = glob(os.path.join(images_dir, "*.jpg")
                     )  # Adjust extension as needed
    invalid_files = []
    for img_file in all_files:
        img = cv2.imread(img_file)
        if img is None:
            invalid_files.append(img_file)
    return invalid_files


invalid_files = validate_dataset(train_images_dir)
if invalid_files:
    print("Invalid images found:", invalid_files)
else:
    print("All images are valid.")
