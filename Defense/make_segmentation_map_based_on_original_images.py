# CODE FROM ChatGPT WITH MODIFICATIONS


import os
import numpy as np
import cv2
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Function to generate a binary mask from comparing two images
def generate_patch_mask(original_image, patched_image, threshold=50):
    diff = cv2.absdiff(original_image, patched_image)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_BINARY)
    
    return binary_mask

# Function to go through the directories and compare images
def compare_folders(original_folder, patched_folder, output_folder, threshold=0):
    original_images = os.listdir(original_folder)
    
    for image_name in original_images:
        # Construct the file paths for the original and patched images
        original_image_path = os.path.join(original_folder, image_name)
        patched_image_path = os.path.join(patched_folder, image_name)
        
        if os.path.exists(patched_image_path):
            # Read the original and patched images
            original_image = cv2.imread(original_image_path)
            patched_image = cv2.imread(patched_image_path)
            
            # Check if the images have the same shape
            if original_image.shape != patched_image.shape:
                print(f"Skipping {image_name} - shapes don't match!")
                continue

            # Generate the binary mask indicating the patch location
            binary_mask = generate_patch_mask(original_image, patched_image, threshold)

            # Resize binary mask to the same height as the original image (to match the left and right image sizes)
            # binary_mask_resized = cv2.resize(binary_mask, (original_image.shape[1], original_image.shape[0]))

            # Stack the original image and the binary mask side by side
            # Save the combined image to the output folder
            mask_output_path = os.path.join(output_folder, f"mask_{image_name}")
            cv2.imwrite(mask_output_path, binary_mask)
            
            print(f"Saved mask image: {image_name}")
        else:
            print(f"Image {image_name} not found in patched folder.")

# Define the paths to the original and patched image folders
original_folder = 'Defense/trainingdata/original'
patched_folder = 'Defense/trainingdata/patched'
mask_folder = 'Defense/trainingdata/segmentationmap'

# Ensure output folder exists
os.makedirs(mask_folder, exist_ok=True)

# Run the comparison
compare_folders(original_folder, patched_folder, mask_folder)
