"""
Adversarial Patch Attack on the GTSRB dataset.

Reference:
[1] Tom B. Brown, Dandelion Mané, Aurko Roy, Martín Abadi, Justin Gilmer
    Adversarial Patch. arXiv:1712.09665
"""
import argparse
import os
import csv
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from patch_utils import *
from utils import *
from GTSRB_CNN.train import get_model, r2_keras


def preprocess_and_denormalize(image):
    """
    Preprocess and denormalize an image for visualization.
    Args:
        image: A TensorFlow tensor or NumPy array representing the image.
               The values should be in [0, 1] (normalized format).
    Returns:
        A NumPy array with pixel values in [0, 255] (uint8 format).
    """
    # Convert tensor to NumPy array if necessary
    if isinstance(image, tf.Tensor):
        image = image.numpy()

    # Squeeze unnecessary dimensions (e.g., batch size)
    image = np.squeeze(image)

    # Ensure values are in the range [0, 1] before scaling
    if image.max() <= 1.0:
        image = image * 255.0

    # Clip to ensure valid pixel range and convert to uint8
    image = np.clip(image, 0, 255).astype(np.uint8)

    return image


def visualize_patch_effect(image, patched_image, idx):
    """
    Visualize (or save) the effect of an adversarial patch on an image.
    This example simply saves them as .png files using cv2.
    """
    # Denormalize images for saving
    orig = preprocess_and_denormalize(image)
    patched = preprocess_and_denormalize(patched_image)

    os.makedirs("output", exist_ok=True)
    # Save the original and patched images
    cv2.imwrite(f"output/original_{idx}.png", orig)
    cv2.imwrite(f"output/patched_{idx}.png", patched)


def patch_attack(image, applied_patch, mask, target, probability_threshold, model, lr=1, max_iteration=100):
    """
    Perform a patch attack via iterative gradient-based optimization.

    Args:
        image (ndarray): original image, shape (H, W, C) in [0..1].
        applied_patch (ndarray): initial adversarial patch, same shape (H, W, C).
        mask (ndarray): binary mask for patch placement, shape (H, W, C).
        target (int): target class index for the adversarial attack.
        probability_threshold (float): threshold for target class probability.
        model (tf.keras.Model): TensorFlow model with classification head.
        lr (float): learning rate for patch optimization.
        max_iteration (int): maximum number of optimization steps.

    Returns:
        (perturbed_image, final_patch): the patched image and the learned patch arrays.
    """
    # Convert arrays to tensors
    image_tf = tf.convert_to_tensor(image, dtype=tf.float32)
    patch_tf = tf.convert_to_tensor(applied_patch, dtype=tf.float32)
    mask_tf = tf.convert_to_tensor(mask, dtype=tf.float32)

    # Expand dims to have batch of size 1
    image_tf = tf.expand_dims(image_tf, axis=0)
    patch_tf = tf.expand_dims(patch_tf, axis=0)
    mask_tf = tf.expand_dims(mask_tf, axis=0)

    target_probability = 0
    iteration = 0
    perturbed_image_tf = None

    # Iterative optimization
    while target_probability < probability_threshold and iteration < max_iteration:
        iteration += 1
        with tf.GradientTape() as tape:
            tape.watch(patch_tf)

            # Apply patch
            perturbed_image_tf = tf.multiply(mask_tf, patch_tf) + tf.multiply((1 - mask_tf), image_tf)
            perturbed_image_tf = tf.clip_by_value(perturbed_image_tf, 0, 1)

            # Forward pass
            classification_output, _ = model(perturbed_image_tf, training=False)

            # We want to maximize the log-softmax of the target class
            log_softmax_output = tf.nn.log_softmax(classification_output, axis=1)
            target_log_softmax = log_softmax_output[0, target]

        # Compute gradient w.r.t patch
        patch_grad = tape.gradient(target_log_softmax, patch_tf)

        # Normalize gradient
        patch_grad = patch_grad / (tf.norm(patch_grad) + 1e-7)

        # Update the patch (gradient ascent)
        patch_tf = patch_tf + lr * patch_grad
        patch_tf = tf.clip_by_value(patch_tf, 0, 1)

        # Check target probability
        classification_output, _ = model(perturbed_image_tf, training=False)
        softmax_output = tf.nn.softmax(classification_output, axis=1)
        target_probability = softmax_output[0, target].numpy()

    # Convert final tensors back to numpy
    perturbed_image = perturbed_image_tf.numpy()[0]
    final_patch = patch_tf.numpy()[0]

    return perturbed_image, final_patch


def main():
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help="batch size")
    parser.add_argument('--train_size', type=int, default=30, help="number of training images to attack")
    parser.add_argument('--test_size', type=int, default=30, help="number of test images to evaluate")
    parser.add_argument('--noise_percentage', type=float, default=0.1, help="patch noise relative to image size")
    parser.add_argument('--probability_threshold', type=float, default=0.9, help="target probability threshold")
    parser.add_argument('--lr', type=float, default=1, help="learning rate for patch optimization")
    parser.add_argument('--max_iteration', type=int, default=1000, help="max optimization iteration")
    parser.add_argument('--target', type=int, default=30, help="target label index")
    parser.add_argument('--epochs', type=int, default=1, help="number of epochs for patch training loop")
    parser.add_argument('--patch_type', type=str, default='rectangle', help="type of patch shape")
    parser.add_argument('--GPU', type=str, default='0', help="index of GPU to use")
    parser.add_argument('--log_dir', type=str, default='train.csv', help='path to log CSV file')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU

    # Load the model
    model = get_model((100, 100))
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        optimizer="adam",
        loss={"classification": loss, "regression": "mse"},
        metrics={"classification": "acc", "regression": r2_keras},
        loss_weights={"classification": 5, "regression": 1}
    )
    weights_file_name = 'result.weights.h5'
    weights_path = f'../GTSRB_CNN/{weights_file_name}'
    model.load_weights(weights_path)
    tqdm.write(f'Model loaded with weights from "{weights_file_name}"')

    # Load the datasets
    train_images = np.load("../data/train.npy")
    test_images = np.load("../data/test.npy")

    # Initialize the patch
    patch = patch_initialization(
        patch_type=args.patch_type,
        image_size=(100, 100, 3),
        noise_percentage=args.noise_percentage
    )
    tqdm.write("Adversarial patch initialized.")

    # Prepare CSV logging
    with open(args.log_dir, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["ClassID", "Original_pred", "Patched_pred", "Original_path", "Patched_path"])

    best_patch_epoch, best_patch_success_rate = 0, 0

    # Lists to store success rates over epochs (for plotting)
    train_success_rates = []
    test_success_rates = []

    for epoch in range(args.epochs):
        tqdm.write(f"=== Epoch {epoch} ===")
        train_success = 0
        train_total = 0

        saved_example = False

        # Use tqdm for a progress bar over the training images
        for idx in tqdm(range(len(train_images)), desc="Training images"):
            image = train_images[idx]
            image = image.astype(np.float32)

            # Forward pass (original) to get the current prediction
            image_tf = tf.expand_dims(image, axis=0)
            classification_output, _ = model(image_tf, training=False)
            original_prediction = tf.argmax(classification_output, axis=1).numpy()[0]

            # Check if the image is not already predicted as the target
            if original_prediction != args.target:
                train_total += 1

                # Generate patch and mask
                applied_patch, mask, x_loc, y_loc = mask_generation(
                    args.patch_type, patch, (100, 100, 3)
                )

                # Run the patch optimization
                perturbated_image, final_patch = patch_attack(
                    image=image,
                    applied_patch=applied_patch,
                    mask=mask,
                    target=args.target,
                    probability_threshold=args.probability_threshold,
                    model=model,
                    lr=args.lr,
                    max_iteration=args.max_iteration
                )

                # Check the new prediction
                perturbed_image_tf = tf.expand_dims(perturbated_image, axis=0)
                classification_output, _ = model(perturbed_image_tf, training=False)
                patched_prediction = tf.argmax(classification_output, axis=1).numpy()[0]

                # Count success if patched image is now predicted as target
                if patched_prediction == args.target:
                    train_success += 1

                    # Only save one example (original vs patched) per epoch
                    if not saved_example:
                        visualize_patch_effect(
                            image=image_tf,
                            patched_image=perturbed_image_tf,
                            idx=f"{epoch}_{idx}"
                        )
                        saved_example = True

                    # Log to CSV
                    with open(args.log_dir, 'a', newline='') as f:
                        writer = csv.writer(f)
                        original_path = f"output/original_{idx}.png"
                        patched_path = f"output/patched_{idx}.png"
                        writer.writerow([idx, original_prediction, patched_prediction, original_path, patched_path])

                    # Update the global patch from the final patch region
                    ph, pw, _ = patch.shape
                    patch = final_patch[x_loc:x_loc + ph, y_loc:y_loc + pw, :]

        # Print success rate for this epoch
        epoch_success_rate = (train_success / train_total) * 100 if train_total > 0 else 0.0
        tqdm.write(f"Epoch {epoch}: Patch attack success rate on train subset: {epoch_success_rate:.2f}%")
        train_success_rates.append(epoch_success_rate)

        # Evaluate on the test set with the current patch
        test_success = 0
        for idx in tqdm(range(len(test_images)), desc="Testing images"):
            test_image, rois, test_label = test_images

            applied_patch, mask, _, _ = mask_generation(
                args.patch_type, patch, (100, 100, 3)
            )

            perturbated_image, _ = patch_attack(
                image=test_image,
                applied_patch=applied_patch,
                mask=mask,
                target=args.target,
                probability_threshold=args.probability_threshold,
                model=model,
                lr=args.lr,
                max_iteration=args.max_iteration
            )

            perturbed_image_tf = tf.expand_dims(perturbated_image, axis=0)
            classification_output, _ = model(perturbed_image_tf, training=False)
            patched_prediction = tf.argmax(classification_output, axis=1).numpy()[0]

            if patched_prediction != test_label:
                test_success += 1

        test_rate = (test_success / len(test_images)) * 100
        tqdm.write(f"Epoch {epoch}: Patch attack success rate on test subset (fooling classifier): {test_rate:.2f}%")

        # Log generation or analytics if desired
        log_generation(args.log_dir)

        # Track best patch if needed
        if test_rate > best_patch_success_rate:
            best_patch_success_rate = test_rate
            best_patch_epoch = epoch
            # Save best patch if needed

    print(
        f"Best patch found at epoch {best_patch_epoch} "
        f"with success rate {best_patch_success_rate:.2f}% on the test set."
    )


if __name__ == '__main__':
    main()
