# Adversarial Patch: utils
# Utils in need to generate the patch and test on the dataset.


import numpy as np
import csv
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the datasets
# We randomly sample some images from the dataset, because ImageNet itself is too large.

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def dataloader( data_dir, batch_size, train_size = 0.8):
    # Define preprocessing transformations for training and testing datasets
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,  # Normalize pixel values to [0, 1]
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.2,
        validation_split=1 - train_size  # Make sure to split training/validation
    )

    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255,  # Normalize pixel values to [0, 1]
        validation_split=1 - train_size  # Make sure to split training/validation
    )


    # Create train and test generators
    train_generator = train_datagen.flow_from_directory(
        directory=data_dir,
        target_size=(32, 32),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,  # Use provided indices for sampling
        subset="training"  # If you use a validation split, configure it here
    )

    test_generator = test_datagen.flow_from_directory(
        directory=data_dir,
        target_size=(32, 32),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
        subset="validation"
    )

    # Wrap generators in TensorFlow Dataset objects for more flexibility
    train_dataset = tf.data.Dataset.from_generator(
        lambda: train_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, 32, 32, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, train_generator.num_classes), dtype=tf.float32)
        )
    )

    test_dataset = tf.data.Dataset.from_generator(
        lambda: test_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, 32, 32, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, test_generator.num_classes), dtype=tf.float32)
        )
    )

    return train_dataset.prefetch(tf.data.AUTOTUNE), test_dataset.prefetch(tf.data.AUTOTUNE)

# Test the model on clean dataset
def test(model, dataloader):
    model.eval()
    correct, total, loss = 0, 0, 0
    with torch.no_grad():
        for (images, labels) in dataloader:
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.shape[0]
            correct += (predicted == labels).sum().item()
    return correct / total

# Load the log and generate the training line
def log_generation(log_dir):
    # Initialize lists to store data
    epochs, train_rate, test_rate = [], [], []
    
    with open(log_dir, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)  # Skip the header row explicitly
        for row in reader:
            # Skip malformed rows
            if len(row) < 3:
                print(f"Skipping malformed row: {row}")
                continue
            # Parse the data
            try:
                epochs.append(int(row[0]))
                train_rate.append(float(row[1]))
                test_rate.append(float(row[2]))
            except ValueError as e:
                print(f"Skipping row due to parsing error: {row}. Error: {e}")

    # Generate the success line
    plt.figure(num=0)
    plt.plot(epochs, test_rate, label='test_success_rate', linewidth=2, color='r')
    plt.plot(epochs, train_rate, label='train_success_rate', linewidth=2, color='b')
    plt.xlabel("epoch")
    plt.ylabel("success rate")
    plt.xlim(-1, max(epochs) + 1)
    plt.ylim(0, 1.0)
    plt.title("patch attack success rate")
    plt.legend()
    plt.savefig("training_pictures/patch_attack_success_rate.png")
    plt.close(0)