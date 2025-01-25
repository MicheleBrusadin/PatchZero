import os
import shutil
import random
def check_folder_empty(folder_path):
    """Check if a folder is empty. Return False if not empty, True otherwise."""
    if os.path.exists(folder_path) and any(os.scandir(folder_path)):
        print(f"Warning: The folder '{folder_path}' is not empty. Please delete it before running the script.")
        return False
    return True

def distribute_images(patched_folder, segmentation_folder, output_folder, test_percentage=70, validate_percentage=15, train_percentage=15):
    # Ensure percentages sum up to 100
    if test_percentage + validate_percentage + train_percentage != 100:
        raise ValueError("The percentages must sum up to 100.")
    
    # Get the list of images in the patched folder
    patched_images = os.listdir(patched_folder)
    
    # Shuffle the images for randomness
    random.shuffle(patched_images)

    if not check_folder_empty(output_folder):
        print("Exiting script to avoid overwriting existing files.")
        return
    
    # Calculate the number of images for each split
    total_images = len(patched_images)
    num_test = int(total_images * test_percentage / 100)
    num_validate = int(total_images * validate_percentage / 100)
    num_train = total_images - (num_test + num_validate)  # Remaining for training
    
    # Create output directories
    for split in ['test', 'validate', 'train']:
        os.makedirs(os.path.join(output_folder, split, 'patched'), exist_ok=True)
        os.makedirs(os.path.join(output_folder, split, 'segmentation'), exist_ok=True)
    
    # Helper function to move images into their respective folders
    def copy_images(image_list, split):
        for image_name in image_list:
            # Paths for patched and segmentation images
            patched_image_path = os.path.join(patched_folder, image_name)
            segmentation_image_path = os.path.join(segmentation_folder, f"mask_{image_name}")
            # os.path.join(output_folder, f"mask_{image_name}")
            
            # Ensure the corresponding segmentation image exists
            if os.path.exists(segmentation_image_path):
                # Destination paths
                patched_dest = os.path.join(output_folder, split, 'patched', image_name)
                segmentation_dest = os.path.join(output_folder, split, 'segmentation', image_name)
                
                # Copy images to the split folder
                shutil.copy(patched_image_path, patched_dest)
                shutil.copy(segmentation_image_path, segmentation_dest)
                print(f"Copied: {image_name} to {split} (patched & segmentation)")
            else:
                print(f"Warning: Corresponding segmentation map for {image_name} not found.")
    
    # Distribute images to test, validate, and train
    copy_images(patched_images[:num_test], 'test')
    copy_images(patched_images[num_test:num_test + num_validate], 'validate')
    copy_images(patched_images[num_test + num_validate:], 'train')
    
    print(f"Distribution complete: {num_test} images in test, {num_validate} in validate, {num_train} in train.")

if __name__ == "__main__":
    # Define input folders
    patched_folder = 'Defense/trainingdata/patched'
    segmentation_folder = 'Defense/trainingdata/segmentationmap'
    output_folder = 'Defense/trainingdata/splitdata'
    
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Define percentages
    test_percentage = 15
    validate_percentage = 15
    train_percentage = 70
    
    # Distribute the images
    distribute_images(patched_folder, segmentation_folder, output_folder, test_percentage, validate_percentage, train_percentage)
