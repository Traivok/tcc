import os

import cv2
import numpy as np


def crop_yolo_objects(image_dir, label_dir, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get all label files
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

    for label_file in label_files:
        # Extract the image file name
        # Adjust if your images have a different extension
        image_file = label_file.replace('.txt', '.jpg')
        image_path = os.path.join(image_dir, image_file)

        # Skip if image file doesn't exist
        if not os.path.exists(image_path):
            print(f"Image file {image_path} not found. Skipping.")
            continue

        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image {image_path}. Skipping.")
            continue

        height, width, _ = image.shape

        # Read the label file
        with open(os.path.join(label_dir, label_file), 'r') as f:
            lines = f.readlines()

        for idx, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"Invalid annotation format in {
                      label_file}: {line}. Skipping.")
                continue

            class_id, x_center, y_center, bbox_width, bbox_height = map(
                float, parts)

            # Convert normalized YOLO coordinates to pixel coordinates
            x_center *= width
            y_center *= height
            bbox_width *= width
            bbox_height *= height

            x_min = int(x_center - bbox_width / 2)
            y_min = int(y_center - bbox_height / 2)
            x_max = int(x_center + bbox_width / 2)
            y_max = int(y_center + bbox_height / 2)

            # Ensure the bounding box is within the image boundaries
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(width, x_max)
            y_max = min(height, y_max)

            # Crop the object from the image
            cropped_object = image[y_min:y_max, x_min:x_max]

            # Save the cropped object image
            output_file = os.path.join(
                output_dir, f"{os.path.splitext(image_file)[0]}_object{idx}.jpg")
            cv2.imwrite(output_file, cropped_object)

        print(f"Processed {label_file} and saved cropped objects to {
              output_dir}.")


if __name__ == "__main__":
    cwd = os.getcwd()
    data_path = os.path.join(cwd, 'ships_v10i')

    # Create directories for train data
    train_images_dir = os.path.join(data_path, "train/images")
    train_labels_dir = os.path.join(data_path, "train/labels")

    output_dir = os.path.join(data_path, "cropped")

    # Run the cropping function
    crop_yolo_objects(train_images_dir, train_labels_dir, output_dir)
