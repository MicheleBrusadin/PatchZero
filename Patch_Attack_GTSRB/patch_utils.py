# Adversarial Patch: patch_utils
# utils for patch initialization and mask generation

import tensorflow as tf
import numpy as np


# Initialize the patch
def patch_initialization(patch_type='rectangle', image_size=(30, 30, 3), noise_percentage=0.1):
    if patch_type != 'rectangle':
        raise ValueError(f"Invalid patch type: {patch_type}")

    mask_length = int((noise_percentage * image_size[0] * image_size[1]) ** 0.5)
    # print("Mask length:", mask_length)
    patch = np.random.rand(mask_length, mask_length, image_size[2]).astype(np.float32)
    return patch


# Generate the mask and apply the patch
def mask_generation(mask_type='rectangle', patch=None, image_size=(100, 100, 3), rotate_patch=False):
    """
    Generate a patch placed at a random or deterministic location on the image.
    Returns:
      applied_patch: (H, W, C) array with the patch positioned somewhere on the image.
      mask: (H, W, C) binary mask with 1s where patch is placed.
      x_location, y_location: top-left coordinates of the patch region.
    """
    if mask_type != 'rectangle':
        raise ValueError(f"Invalid mask type: {mask_type}")

    # Create a blank (zero) array for the applied patch with the same shape as the image
    applied_patch = np.zeros(image_size, dtype=patch.dtype)

    # Rotate the patch randomly
    if rotate_patch:
        rotated_patch = np.copy(patch)
        rotation_angle = np.random.choice(4)
        for i in range(rotated_patch.shape[2]):
            rotated_patch[..., i] = np.rot90(rotated_patch[..., i], rotation_angle)
        patch = rotated_patch

    ph, pw, pc = patch.shape
    ih, iw, ic = image_size

    # Ensure the patch does not exceed the image dimensions
    # If patch is bigger, raise an error or handle it:
    if ph > ih or pw > iw:
        raise ValueError(f"Patch size ({ph}x{pw}) cannot exceed image size ({ih}x{iw}).")

    # If the patch exactly matches the image, there's only one possible location: top-left corner (0, 0).
    # Otherwise, pick a random location so that the patch is fully within the image.
    max_x = ih - ph
    max_y = iw - pw
    if max_x == 0 and max_y == 0:
        # Patch is exactly the image size
        x_location, y_location = 0, 0
    else:
        # Random valid coordinate for patch
        x_location = np.random.randint(0, ih - ph)
        y_location = np.random.randint(0, iw - pw)

    # Apply the patch to the blank image at the chosen location
    applied_patch[x_location:x_location + ph, y_location:y_location + pw, :] = patch

    # Create a mask in the same way
    # mask = np.zeros_like(applied_patch)
    mask = np.zeros(image_size, dtype=patch.dtype)
    mask[x_location:x_location + ph, y_location:y_location + pw, :] = 1.0

    return applied_patch, mask, x_location, y_location


# Test the patch on dataset


def test_patch(patch_type, target, patch, test_loader, model):
    """
    Evaluate the success rate of an adversarial patch on the test dataset.

    Args:
        patch_type: Type of patch (e.g., "rectangle").
        target: Target label for the adversarial attack.
        patch: Initial adversarial patch (NumPy array).
        test_loader: TensorFlow Dataset or generator for test data.
        model: TensorFlow model to test.

    Returns:
        Success rate of the adversarial patch.
    """
    test_total = 0
    test_actual_total = 0
    test_success = 0

    for image, label, *_ in test_loader:
        # Ensure only one image is processed at a time
        assert image.shape[0] == 1, "Only one image should be loaded at a time."

        # Forward pass for the clean image
        classification_output, detection_output = model(image, training=False)
        predicted = tf.argmax(classification_output, axis=1)

        # Skip if the model's clean prediction matches the target ((or the label))
        if predicted.numpy()[0] != target:  # and predicted.numpy()[0] != tf.argmax(label, axis=1).numpy()[0]
            test_actual_total += 1

            # Generate the patch and mask
            applied_patch, mask, x_location, y_location = mask_generation(patch_type, patch, image_size=(100, 100, 3))

            # Apply the patch to the image
            perturbated_image = tf.multiply(mask, applied_patch) + tf.multiply(1 - mask, image)
            perturbated_image = tf.clip_by_value(perturbated_image, 0, 1)

            # Forward pass for the patched image
            classification_output, detection_output = model(perturbated_image, training=False)
            predicted = tf.argmax(classification_output, axis=1)

            # Check if the patched prediction matches the target label
            if predicted.numpy()[0] == target:
                test_success += 1

    return test_success / test_actual_total if test_actual_total > 0 else 0
