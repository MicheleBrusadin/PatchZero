import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (Convolution2D, BatchNormalization, ReLU, LeakyReLU, Add, Activation,
                                     GlobalAveragePooling2D, AveragePooling2D, UpSampling2D)
from tqdm import tqdm


def ensure_directory_exists(directory):
    """
    Ensures that the specified directory exists. If not, it creates it.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_image_mask_subset(path):
    """
    Loads images from two subdirectories: "patched" and "segmentation". Normalizes them,
    converts the color channels from BGR to RGB, and returns them along with the total number of samples
    and the file names of the patched images.

    :param path: The base directory containing 'patched' and 'segmentation' subdirectories
    :return: (image_list, mask_list, num_samples, file_name_patch)
             image_list: List of loaded and normalized images (patches)
             mask_list:  List of loaded and normalized masks
             num_samples: Number of mask samples found
             file_name_patch: List of filenames in the 'patched' folder
    """
    file_name_patch = os.listdir(os.path.join(path, "images"))
    file_name_mask = os.listdir(os.path.join(path, "masks"))

    num_samples = len(file_name_mask)

    image_list, mask_list = [], []

    # Load the patched images
    for file_name in tqdm(file_name_patch, desc=f"Loading patched images from {path}"):
        image = cv2.imread(os.path.join(path, "images", file_name))
        image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
        image = image[:, :, ::-1]  # BGR to RGB
        image_list.append(image)
        del image

    # Load the segmentation masks
    for file_name in tqdm(file_name_mask, desc=f"Loading segmentation masks from {path}"):
        mask = cv2.imread(os.path.join(path, "masks", file_name))
        mask = cv2.normalize(mask, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
        mask_list.append(mask)
        del mask

    return image_list, mask_list, num_samples, file_name_patch


def convolutional_block(input_tensor, filters, block_identifier):
    """
   Builds a convolutional residual block with three convolution sub-blocks and skip connection.
   Uses dilation in the second sub-block.

   :param input_tensor: Input feature map (tensor)
   :param filters: Tuple of filter sizes (filter1, filter2, filter3)
   :param block_identifier: A unique identifier (string or int) for naming the block layers
   :return: The output tensor after applying the dilated convolutional block
    """
    # Dilated convolution block
    block_name = f'block_{block_identifier}_'
    filter1, filter2, filter3 = filters
    skip_connection = input_tensor

    # Block A
    input_tensor = Convolution2D(filters=filter1, kernel_size=(1, 1), dilation_rate=(1, 1),
                                 padding='same', kernel_initializer='he_normal', name=block_name + 'a')(input_tensor)
    input_tensor = BatchNormalization(name=block_name + 'batch_norm_a')(input_tensor)
    input_tensor = LeakyReLU(alpha=0.2, name=block_name + 'leakyrelu_a')(input_tensor)

    # Block B
    input_tensor = Convolution2D(filters=filter2, kernel_size=(3, 3), dilation_rate=(2, 2),
                                 padding='same', kernel_initializer='he_normal', name=block_name + 'b')(input_tensor)
    input_tensor = BatchNormalization(name=block_name + 'batch_norm_b')(input_tensor)
    input_tensor = LeakyReLU(alpha=0.2, name=block_name + 'leakyrelu_b')(input_tensor)

    # Block C
    input_tensor = Convolution2D(filters=filter3, kernel_size=(1, 1), dilation_rate=(1, 1),
                                 padding='same', kernel_initializer='he_normal', name=block_name + 'c')(input_tensor)
    input_tensor = BatchNormalization(name=block_name + 'batch_norm_c')(input_tensor)

    # Skip convolutional block for residual
    skip_connection = Convolution2D(filters=filter3, kernel_size=(3, 3), padding='same', name=block_name + 'skip_conv')(
        skip_connection)
    skip_connection = BatchNormalization(name=block_name + 'batch_norm_skip_conv')(skip_connection)

    # Block C + Skip Convolution
    input_tensor = Add(name=block_name + 'add')([input_tensor, skip_connection])
    input_tensor = ReLU(name=block_name + 'relu')(input_tensor)

    return input_tensor


def base_convolutional_block(input_layer, image_size):
    """
    Applies a series of three convolutional_block calls to extract base features
    from the input layer.

    :param input_layer: Input image or feature map
    :param image_size: Parameter used to configure the number of filters
    :return: Output feature map after three convolutional_block calls
    """
    # Base Block 1
    base_result = convolutional_block(input_layer, [int(image_size / 8), int(image_size / 8), int(image_size / 4)], '1')
    # Base Block 2
    base_result = convolutional_block(base_result, [int(image_size / 4), int(image_size / 4), int(image_size / 2)], '2')
    # Base Block 3
    base_result = convolutional_block(base_result, [int(image_size / 2), int(image_size / 2), image_size], '3')

    return base_result


def pyramid_pooling_module(input_layer, image_size):
    """
    Creates a pyramid pooling module by combining the output of the base
    convolutional block with several pooled feature maps at different scales.

    :param input_layer: The input tensor
    :param image_size: Used to determine the dimensionality of certain layers
    :return: Concatenated feature map from base features and multiple pooled scales
    """
    base_result = base_convolutional_block(input_layer, image_size)

    # Red Pixel Pooling
    red_result = GlobalAveragePooling2D(name='red_pool')(base_result)
    red_result = tf.keras.layers.Reshape((1, 1, image_size))(red_result)
    red_result = Convolution2D(filters=64, kernel_size=(1, 1), name='red_1_by_1')(red_result)
    red_result = UpSampling2D(size=image_size, interpolation='bilinear', name='red_upsampling')(red_result)

    # Yellow Pixel Pooling
    yellow_result = AveragePooling2D(pool_size=(2, 2), name='yellow_pool')(base_result)
    yellow_result = Convolution2D(filters=64, kernel_size=(1, 1), name='yellow_1_by_1')(yellow_result)
    yellow_result = UpSampling2D(size=2, interpolation='bilinear', name='yellow_upsampling')(yellow_result)

    # Blue Pixel Pooling
    blue_result = AveragePooling2D(pool_size=(4, 4), name='blue_pool')(base_result)
    blue_result = Convolution2D(filters=64, kernel_size=(1, 1), name='blue_1_by_1')(blue_result)
    blue_result = UpSampling2D(size=4, interpolation='bilinear', name='blue_upsampling')(blue_result)

    # Green Pixel Pooling
    green_result = AveragePooling2D(pool_size=(8, 8), name='green_pool')(base_result)
    green_result = Convolution2D(filters=64, kernel_size=(1, 1), name='green_1_by_1')(green_result)
    green_result = UpSampling2D(size=8, interpolation='bilinear', name='green_upsampling')(green_result)

    # Concatenate all pooled features
    return tf.keras.layers.concatenate([base_result, red_result, yellow_result, blue_result, green_result])


def pyramid_based_conv(input_layer, image_size):
    """
    Creates a final output layer from a pyramid pooling module.
    Outputs a 3-channel feature map with sigmoid activation.

    :param input_layer: The input tensor
    :param image_size: Used to build the pyramid pooling module
    :return: The final output tensor with 3 channels (sigmoid activation)
    """
    result = pyramid_pooling_module(input_layer, image_size)
    result = Convolution2D(filters=3, kernel_size=3, padding='same', name='last_conv_3_by_3')(result)
    result = BatchNormalization(name='last_conv_3_by_3_batch_norm')(result)
    result = Activation('sigmoid', name='last_conv_relu')(result)

    return result


def plot_and_save_comparison_figure(img, mask, pred, image_size, save_path=None):
    """
    Plots the original image, the mask, and the predicted mask side-by-side.
    Optionally saves the figure if save_path is provided.

    :param img: Original image
    :param mask: Ground truth mask
    :param pred: Model-predicted mask
    :param image_size: Height/width of the image
    :param save_path: If provided, the figure is saved to this path instead of being displayed
    """
    mask = np.reshape(mask, (image_size, image_size, 3))
    pred = np.reshape(pred, (image_size, image_size, 3))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 10))
    ax1.imshow(img)
    ax1.set_title('Original image')
    ax2.imshow(mask)
    ax2.set_title('Original mask')
    ax3.imshow(pred)
    ax3.set_title('Predicted mask')

    plt.savefig(save_path)
    plt.close(fig)


def interval_mapping(image, from_min, from_max, to_min, to_max):
    """
    Maps an image's pixel intensities from one range to another.

    :param image: The input image (NumPy array)
    :param from_min: Minimum of the current range
    :param from_max: Maximum of the current range
    :param to_min: Minimum of the new target range
    :param to_max: Maximum of the new target range
    :return: Image with rescaled pixel intensities
    """
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    return to_min + (scaled * to_range)


def create_model(input_shape, image_size):
    """
    Creates and returns a Keras Model that applies a pyramid-based convolution architecture.

    :param input_shape: Shape of the input images (height, width, channels)
    :param image_size: Used to build the pyramid pooling module
    :return: A compiled Keras model
    """
    input_layer = tf.keras.Input(shape=input_shape, name='input')
    output_layer = pyramid_based_conv(input_layer, image_size)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model


def train_model(model, train_images, train_masks, batch_size, epochs):
    """
    Compiles and trains the model using MSE loss and the Adam optimizer.

    :param model: A Keras model to train
    :param train_images: Numpy array of training images
    :param train_masks: Numpy array of training masks (labels)
    :param batch_size: Batch size for training
    :param epochs: Number of training epochs
    :return: History object from Keras model.fit()
    """
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

    history = model.fit(
        train_images,
        train_masks,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    return history


def save_training_loss_plot(history, save_path):
    """
    Plots the training loss curve from the Keras history object.
    Saves the plot to the specified path.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def threshold_masks(pred_masks, threshold_value):
    """
    Applies a threshold to a list of predicted masks, creating binary masks.

    :param pred_masks: List or array of predicted masks
    :param threshold_value: Threshold above which pixel is set to 1, else 0
    :return: List of binary masks
    """
    binary_masks = []
    # Using tqdm to display progress in thresholding
    for mask in tqdm(pred_masks, desc="Applying threshold to predicted masks"):
        _, binary_image = cv2.threshold(mask, threshold_value, 1, cv2.THRESH_BINARY)
        binary_masks.append(binary_image)
    return binary_masks


def dilate_masks(binary_masks, kernel_size=10, iterations=1):
    """
    Dilates a list of binary masks using a square kernel of given size and returns the result.

    :param binary_masks: List of binary masks
    :param kernel_size: Size of the square kernel used for dilation
    :param iterations: Number of dilation iterations
    :return: List of dilated masks
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = []
    # Loop through each binary mask and dilate it
    for mask in tqdm(binary_masks, desc="Dilating binary masks"):
        dilated_mask = cv2.dilate(mask, kernel, iterations=iterations)
        dilated.append(dilated_mask)
    return dilated


def create_patchzero_images(dilated_masks, original_images):
    """
    Replaces the dilated mask region of each image with the mean pixel value of
    the non-masked region, effectively 'zeroing out' the patch region.

    :param dilated_masks: List of dilated binary masks
    :param original_images: List of original images
    :return: List of 'patch-zeroed' images
    """
    patchzero_images = []
    # Using tqdm to display progress during patch zero creation
    for mask, image in tqdm(zip(dilated_masks, original_images),
                            desc="Creating PatchZero images",
                            total=len(dilated_masks)):
        masked_image = np.ma.masked_where(mask == 1, image)  # ignore masked area
        average_pixel_value = masked_image.mean()            # mean value of the unmasked region
        patchzero = image * (1 - mask) + mask * average_pixel_value
        patchzero_images.append(patchzero)
    return patchzero_images


def save_patchzero_images(patchzero_images, file_names, output_folder):
    """
    Saves the 'patch-zeroed' images to an output folder, using the same filenames
    as the original patched images. The images are de-normalized to [0,255] before saving.

    :param patchzero_images: List of patch-zeroed images
    :param file_names: List of filenames corresponding to each image
    :param output_folder: Directory path where the images will be saved
    """
    # Make sure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Using tqdm to display progress in saving images
    for idx, patchzero in enumerate(tqdm(patchzero_images, desc="Saving PatchZero images")):
        original_image_name = file_names[idx]
        patchzero_image_uint8 = np.clip(patchzero * 255, 0, 255).astype(np.uint8)  # de-normalize
        patchzero_image_uint8 = cv2.cvtColor(patchzero_image_uint8, cv2.COLOR_BGR2RGB)

        output_path = os.path.join(output_folder, original_image_name)
        cv2.imwrite(str(output_path), patchzero_image_uint8)

        # Replace print with tqdm.write to keep output on the same line
        tqdm.write(f"Saved: {original_image_name} at {output_path}")


def defense_main(train_dir="trainingdata/splitdata/train", test_dir="trainingdata/splitdata/test",
                 output_dir="patchzeroimages", plots_dir="plots", model_filename='modelimagenet.keras', batch_size=8,
                 epochs=50):
    """
    Main function that orchestrates the data loading, model creation, training, prediction,
    post-processing (threshold, dilation), creation of PatchZero images, and saving results.
    """
    # Define data folders
    ensure_directory_exists(plots_dir)

    # Load data
    tqdm.write("Loading training data...")
    train_images, train_masks, num_samples_train, file_name_train_patches = load_image_mask_subset(train_dir)
    tqdm.write("Loading test data...")
    test_images, test_masks, num_samples_test, file_name_test_patches = load_image_mask_subset(test_dir)

    # Determine image shape and size
    imageshape = train_images[1].shape
    image_size = imageshape[0]

    # Add noise to patch place of train_images
    noise = np.random.rand(image_size, image_size, 3)
    train_mask_noise = []
    for mask in train_masks:
        mask_noise = mask * noise
        train_mask_noise.append(mask_noise)

    plt.imsave(os.path.join(plots_dir, "train_mask_noise_sample.png"), train_mask_noise[0].reshape(image_size,
                                                                                                   image_size, 3))

    tqdm.write(f"Shape of train_mask_noise: {np.shape(train_mask_noise)}")
    tqdm.write(f"Shape of train_images: {np.shape(train_images)}")

    # Create and train the model
    model = create_model(input_shape=train_images[0].shape, image_size=image_size)
    model.summary()

    tqdm.write("Training the model...")
    history = train_model(
        model,
        np.array(train_images, dtype='float16'),
        np.array(train_mask_noise, dtype='float16'),
        batch_size=batch_size,
        epochs=epochs
    )

    # Save and load the trained model
    model.save(model_filename)
    model = keras.models.load_model(model_filename)
    tqdm.write("Model saved and reloaded.")

    save_training_loss_plot(history, save_path=os.path.join(plots_dir, 'training_loss.png'))

    pred_masks_test = model.predict(np.array(test_images, dtype='float16'))
    threshold_value = 0.7
    pred_binary_masks = threshold_masks(pred_masks_test, threshold_value)
    plt.imsave(os.path.join(plots_dir, "binary_mask_sample.png"), pred_binary_masks[0], cmap="gray")

    dilated_masks = dilate_masks(pred_binary_masks, kernel_size=3, iterations=1)
    plt.imsave(os.path.join(plots_dir, "dilated_mask_sample.png"), dilated_masks[0], cmap="gray")

    patchzero_images = create_patchzero_images(dilated_masks, test_images)
    plt.imsave(os.path.join(plots_dir, "patchzero_sample.png"), patchzero_images[0])

    tqdm.write(f"Shape of patchzero_images: {np.shape(patchzero_images)}")
    tqdm.write(f"Max value in patchzero_images: {np.max(patchzero_images)}")

    save_patchzero_images(patchzero_images, file_name_test_patches, output_dir)


if __name__ == '__main__':
    defense_main()
