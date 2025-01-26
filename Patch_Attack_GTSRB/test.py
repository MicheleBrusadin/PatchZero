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
    image_path = "training_pictures_GTSRB/patched/5.png"
    img_size = (30, 30)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Read the image
    image = (image/255.0)
    # image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    # print("Image Shape:", image.shape)
    # print("Image max value:", np.max(image))
    # print("image", image)

    # Ensure color channels are read correctly
    loaded_array = np.load('patched_image.npy')
    # print("image - array", np.sum(loaded_array > 1))
    # print("image - array", np.sum(image-loaded_array > 1))

    

        # Ensure proper value range before scaling


    # print("Loaded Array:", loaded_array.min(), loaded_array.max())
    # normalized_image = (loaded_array - loaded_array.min()) / (loaded_array.max() - loaded_array.min()) * 255
    # normalized_image = normalized_image.astype(np.uint8)
    # cv2.imwrite("normalized.png", normalized_image)

    # normalized_image = cv2.imread("normalized.png", cv2.IMREAD_COLOR)


    # denormalized_array = (normalized_image/255) * 6 - 3


    


    # image_array = np.clip(loaded_array * 255, 0, 255).astype(np.uint8) #de-normalize
    # image_array= cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    

    # image_array_tensor = tf.convert_to_tensor(denormalized_array, dtype=tf.float32)
    # image_array_tensor = tf.expand_dims(image_array_tensor, axis=0)
    # image = cv2.resize(image, img_size)  # Resize to expected model input size
    image = tf.convert_to_tensor(image, dtype=tf.float32) # Normalize to match model input expectations
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Load the image tensor ( this works)
    raw_tensor = tf.io.read_file('image_tensor.tfrecord')
    loaded_tensortf = tf.io.parse_tensor(raw_tensor, out_type=tf.float32)
    



    
    # print( "array max", np.max(loaded_array))
    
    # print("Loaded Array:", loaded_array)
    # print("Loaded Array Shape:", loaded_array.shape)
    # Convert back to TensorFlow tensor and add batch dimension
    loaded_tensor = tf.convert_to_tensor(loaded_array, dtype=tf.float32)
    loaded_tensor = tf.expand_dims(loaded_tensor, axis=0)
    
    classification_output, detection_output = model(image, training=False)
    patched_prediction = tf.argmax(classification_output , axis=1)
    print("Patched Prediction:", patched_prediction)

    print("Classification Output:", classification_output)
    print("Detection Output:", detection_output)