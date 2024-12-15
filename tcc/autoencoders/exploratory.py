import os
import cv2
import numpy as np
from statistics import mean, median, mode, StatisticsError


def calculate_image_size_stats(folder_path):
    image_sizes = []

    # Iterate through all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # Read the image
            file_path = os.path.join(folder_path, file_name)
            image = cv2.imread(file_path)
            if image is not None:
                height, width = image.shape[:2]
                image_sizes.append((width, height))

    if not image_sizes:
        print("No valid images found in the folder.")
        return

    # Calculate statistics
    widths = [size[0] for size in image_sizes]
    heights = [size[1] for size in image_sizes]

    avg_width, avg_height = mean(widths), mean(heights)
    median_width, median_height = median(widths), median(heights)

    try:
        mode_width, mode_height = mode(widths), mode(heights)
    except StatisticsError:
        mode_width, mode_height = None, None  # No unique mode

    print(f"Average Image Size: {avg_width:.2f}x{avg_height:.2f}")
    print(f"Median Image Size: {median_width}x{median_height}")
    if mode_width is not None and mode_height is not None:
        print(f"Mode Image Size: {mode_width}x{mode_height}")
    else:
        print("Mode Image Size: No unique mode")

    return {
        "average": (avg_width, avg_height),
        "median": (median_width, median_height),
        "mode": (mode_width, mode_height)
    }


if __name__ == "__main__":
    # Replace with the path to your cropped folder
    cwd = os.getcwd()
    data_path = os.path.join(cwd, 'ships_v10i')
    cropped_folder = os.path.join(data_path, "cropped")

    calculate_image_size_stats(cropped_folder)
