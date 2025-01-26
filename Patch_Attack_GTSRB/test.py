if __name__ == "__main__":


    import argparse
    import csv
    
    import numpy as np
    
    from patch_utils import*
    from utils import*
    import pandas as pd
    
    
    import os
    from GTSRB_CNN.train import get_model, r2_keras
    
    import tensorflow as tf
    from tensorflow.keras.losses import SparseCategoricalCrossentropy
    from PIL import Image

    

    model = get_model()
    loss = SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer="adam", loss={"classification": loss, "regression": "mse"},
                    metrics={"classification": "acc", "regression": r2_keras},
                    loss_weights={"classification": 5, "regression": 1})
    model.load_weights("weights/weights.h5")
    print("Model is loaded")
    image_path = "training_pictures_GTSRB\patched\epoch_0_sample_4.png"
    img_size = (30, 30)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Read the image
    # Ensure color channels are read correctly
    # image = cv2.resize(image, img_size)  # Resize to expected model input size
    image = tf.convert_to_tensor(image, dtype=tf.float32) # Normalize to match model input expectations
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    raw_tensor = tf.io.read_file('image_tensor.tfrecord')
    loaded_tensortf = tf.io.parse_tensor(raw_tensor, out_type=tf.float32)
    



    # loaded_array = np.load('patched_image.npy')

    # Convert back to TensorFlow tensor and add batch dimension
    # loaded_tensor = tf.convert_to_tensor(loaded_array, dtype=tf.float32)
    # loaded_tensor = tf.expand_dims(loaded_tensor, axis=0)
    
    classification_output, detection_output = model(image, training=False)
    patched_prediction = tf.argmax(classification_output , axis=1)
    print("Patched Prediction:", patched_prediction)

    print("Classification Output:", classification_output)
    print("Detection Output:", detection_output)