# Adversarial Patch: patch_utils
# utils for patch initialization and mask generation


import numpy as np
import tensorflow as tf
import numpy as np

# Initialize the patch
# TODO: Add circle type
def patch_initialization(patch_type='rectangle', image_size=(30, 30, 3), noise_percentage=0.1):
    if patch_type == 'rectangle':
        mask_length = int((noise_percentage * image_size[0] * image_size[1])**0.5)
        # print("Mask length:", mask_length)
        patch = np.random.rand(mask_length, mask_length,image_size[2]).astype(np.float32)
    return patch

# Generate the mask and apply the patch
# TODO: Add circle type
def mask_generation(mask_type='rectangle', patch=None, image_size=(30, 30, 3)):
    applied_patch = np.zeros(image_size)

    

    if mask_type == 'rectangle':
        # patch rotation
        rotation_angle = np.random.choice(4)
        for i in range(patch.shape[2]):
            patch[...,i] = np.rot90(patch[...,i], rotation_angle)  # The actual rotation angle is rotation_angle * 90
        # patch location
        # print("Patch shape:", patch.shape)
        # print('image_size:', image_size)
        x_location, y_location = np.random.randint(low=0, high=image_size[0]-patch.shape[0]), np.random.randint(low=0, high=image_size[1]-patch.shape[1])
        # print("Patch shape:", patch.shape)  # Should match applied_patch slice shape
        # print("Applied patch target slice shape:", applied_patch[x_location:x_location + patch.shape[0], y_location:y_location + patch.shape[1],:].shape)
        
        applied_patch[ x_location:x_location + patch.shape[0], y_location:y_location + patch.shape[1],:] = patch
    mask = np.zeros_like(applied_patch)
    mask[x_location:x_location + patch.shape[0], y_location:y_location + patch.shape[1], :] = 1.0
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
        if  predicted.numpy()[0] != target:  # and predicted.numpy()[0] != tf.argmax(label, axis=1).numpy()[0] 
            test_actual_total += 1

            # Generate the patch and mask
            applied_patch, mask, x_location, y_location = mask_generation(patch_type, patch, image_size=(30,30,3))
            
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
