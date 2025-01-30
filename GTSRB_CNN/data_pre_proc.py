import os
import cv2
import numpy as np
import pandas as pd
from keras_preprocessing.image import img_to_array
from tqdm import tqdm
import matplotlib.pyplot as plt


def resize_bbox(row, img_size):
    """
    Transform bbox coordinates according to resized image
    """
    # compute scaling factor
    x_scale = img_size[0] / row['Width']
    y_scale = img_size[1] / row['Height']

    # transform upper left box coordinates
    x0 = int(np.round(row['Roi.X1'] * x_scale))
    y0 = int(np.round(row['Roi.Y1'] * y_scale))
    # transform lower right box coordinates
    x1 = int(np.round(row['Roi.X2'] * x_scale))
    y1 = int(np.round(row['Roi.Y2'] * y_scale))

    return x0, y0, x1, y1


def read_data(df, filename, img_size=(30, 30), min_size=None, data_dir='datasets', save_dir='output'):
    """
    Read and preprocess GTSRB images and bboxes and save ndarrays to disk.

    - Skips any image smaller than min_size.
    - Resizes all images to img_size.
    - Tracks the number of skipped, upscaled, and downscaled images per class.
    - Saves the data to disk and produces a bar graph.
    """
    images = []
    bbox = []
    labels = []

    # Dictionary to keep counts per class:
    # { class_id: {'skipped': 0, 'upscaled': 0, 'downscaled': 0} }
    class_stats = {}

    # Prepare a progress bar for iterating over the DataFrame
    # total=df.shape[0] ensures the progress bar length matches the number of rows.
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc=f'Reading {filename}'):
        class_id = row['ClassId']

        # Initialize this class_id in class_stats if not present
        if class_id not in class_stats:
            class_stats[class_id] = {'skipped': 0, 'upscaled': 0, 'no change': 0, 'downscaled': 0}

        original_w = row['Width']
        original_h = row['Height']

        # Check if the image is smaller than min_size. If so, skip.
        if min_size and ((original_w < min_size[0]) or (original_h < min_size[1])):
            class_stats[class_id]['skipped'] += 1
            continue

        # Determine if this will be an upscale or downscale
        if (original_w < img_size[0]) or (original_h < img_size[1]):
            # It's smaller in at least one dimension -> upscaling
            class_stats[class_id]['upscaled'] += 1
        elif (original_w > img_size[0]) or (original_h > img_size[1]):
            # It's larger in at least one dimension -> downscaling
            class_stats[class_id]['downscaled'] += 1
        else:
            # Exactly matches both dimensions -> no change
            class_stats[class_id]['no change'] += 1
            pass

        # read image
        img_path = os.path.join(data_dir, *row['Path'].split('/'))
        img = cv2.imread(img_path)

        # resize image to the specified img_size
        img = cv2.resize(img, img_size)
        images.append(img_to_array(img))

        # resize bbox coordinates
        bbox.append(resize_bbox(row, img_size))

        # Add class_id to labels
        labels.append(class_id)

    # Convert to float ndarrays and normalize image data
    images = np.array(images, dtype='float') / 255.
    bbox = np.array(bbox, dtype='float')
    labels = np.array(labels, dtype='float')

    # Save data to disk
    out_path = os.path.join(save_dir, filename)
    with open(out_path, 'wb') as f:
        np.save(f, images)
        np.save(f, bbox)
        np.save(f, labels)

    # Create bar chart for each class with number of skipped, upscaled, downscaled
    plot_class_stats(class_stats, filename, save_dir)

    return images, bbox, labels


def plot_class_stats(class_stats, filename, save_dir='output'):
    """
    Create and save a stacked bar chart showing the number of skipped, upscaled,
    and downscaled images per class.
    """
    # Sort class IDs to have a consistent x-axis
    classes = sorted(class_stats.keys())

    skipped = [class_stats[c]['skipped'] for c in classes]
    upscaled = [class_stats[c]['upscaled'] for c in classes]
    downscaled = [class_stats[c]['downscaled'] for c in classes]

    x = np.arange(len(classes))  # the label locations

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot stacked bars
    ax.bar(x, skipped, label='Skipped', color='red')
    ax.bar(x, upscaled, bottom=skipped, label='Upscaled', color='green')
    ax.bar(x, downscaled, bottom=np.array(skipped) + np.array(upscaled), label='Downscaled', color='blue')

    ax.set_xlabel('Class ID')
    ax.set_ylabel('Count')
    ax.set_title(f'GTSRB Processing Stats - {filename}')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()

    plt.tight_layout()

    # Save figure
    plot_filename = f'stacked_bargraph_{filename.replace(".npy", ".png")}'
    plt.savefig(os.path.join(save_dir, plot_filename))
    plt.close(fig)


def load_data(path):
    """
    Load ndarrays from disk
    """
    with open(path, 'rb') as f:
        images = np.load(f)
        bbox = np.load(f)
        labels = np.load(f)

    return images, bbox, labels


def data_pre_proc_main(data_dir='../datasets/original/gtsrb', train_csv_filename='Train.csv',
                       test_csv_filename='Test.csv', save_dir='../data',
                       img_size=(100, 100), min_size=(40, 40), train_filename='train.npy', test_filename='test.npy'):
    train = pd.read_csv(f'{data_dir}/{train_csv_filename}')
    test = pd.read_csv(f'{data_dir}/{test_csv_filename}')

    read_data(train, filename=train_filename, img_size=img_size, min_size=min_size,
              data_dir=data_dir, save_dir=save_dir)
    read_data(test, filename=test_filename, img_size=img_size, min_size=min_size,
              data_dir=data_dir, save_dir=save_dir)


if __name__ == '__main__':
    data_pre_proc_main()
