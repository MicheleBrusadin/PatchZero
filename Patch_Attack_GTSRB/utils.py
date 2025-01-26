import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from keras_preprocessing.image import img_to_array
import csv

def resize_bbox(row, img_size):
    """
    Transform bounding box coordinates according to the resized image.
    """
    x_scale = img_size[0] / row["Width"]
    y_scale = img_size[1] / row["Height"]

    # Transform upper left box coordinates
    x0 = int(np.round(row["Roi.X1"] * x_scale))
    y0 = int(np.round(row["Roi.Y1"] * y_scale))

    # Transform lower right box coordinates
    x1 = int(np.round(row["Roi.X2"] * x_scale))
    y1 = int(np.round(row["Roi.Y2"] * y_scale))

    return x0, y0, x1, y1


def preprocess_image_and_bbox(image_path, bbox, img_size=(30, 30)):
    """
    Load an image, resize it, and adjust its bounding box accordingly.
    """
    # Load and resize the image
    image = cv2.imread(image_path)
    image = cv2.resize(image, img_size)
    image = img_to_array(image) / 255.0  # Normalize to [0, 1]

    # Adjust bounding box coordinates
    x0, y0, x1, y1 = resize_bbox(bbox, img_size)

    return image, [x0, y0, x1, y1]


def preprocess_dataset(df, data_dir, img_size=(30, 30)):
    """
    Process the entire dataset to normalize images and bounding boxes.
    Returns processed images, bounding boxes, and labels as arrays.
    """
    images, labels, bboxes = [], [], []

    for _, row in df.iterrows():
        image_path = os.path.join(data_dir, row["Path"])
        label = row["ClassId"]
        bbox = {
            "Roi.X1": row["Roi.X1"],
            "Roi.Y1": row["Roi.Y1"],
            "Roi.X2": row["Roi.X2"],
            "Roi.Y2": row["Roi.Y2"],
            "Width": row["Width"],
            "Height": row["Height"],
        }

        image, bbox_coords = preprocess_image_and_bbox(image_path, bbox, img_size)
        images.append(image)
        labels.append(label)
        bboxes.append(bbox_coords)
        

    return np.array(images), np.array(labels), np.array(bboxes)




def create_dataloader(images, labels, bboxes, batch_size, shuffle=True):
    """
    Create a TensorFlow dataset from preprocessed images, bounding boxes, and labels.
    """

    dataset = tf.data.Dataset.from_tensor_slices((images, labels, bboxes))

    if shuffle:
        dataset = dataset.shuffle(len(images))

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def visualize_image_with_bbox(image, label, bbox, class_names=None):
    """
    Visualize an image with its bounding box and class label.
    """


    plt.imshow(image)
    ax = plt.gca()

    # Draw bounding box
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0
    rect = Rectangle((x0, y0), width, height, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    # Add label
    if class_names:
        plt.text(x0, y0 - 10, class_names[label], color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    else:
        plt.text(x0, y0 - 10, str(label), color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    plt.axis("off")
    plt.show()


# Test the model on clean dataset
def test(model, dataloader):
    """
    Evaluate the model on a given dataset.

    Args:
        model: TensorFlow model to evaluate.
        dataloader: TensorFlow Dataset for evaluation.

    Returns:
        Accuracy: Proportion of correctly predicted labels.
    """
    # Switch the model to evaluation mode
    correct = 0
    total = 0

    # Disable gradient computation for evaluation
    for images, labels, *_ in dataloader:
        total += labels.shape[0]
        print(f"Images shape: {images.shape}, Labels shape: {labels.shape}")
        break
        # Ensure the inputs are on the correct device
        images = tf.convert_to_tensor(images)
        labels = tf.convert_to_tensor(labels)

        # Forward pass
        classification_output, detection_output = model(images, training=False)

        # Get predicted class (argmax across classes)
        predicted = tf.argmax(classification_output, axis=1)

        # Compare with ground-truth labels
        correct += tf.reduce_sum(tf.cast(predicted == tf.argmax(labels, axis=1), tf.float32)).numpy()
        total += labels.shape[0]

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