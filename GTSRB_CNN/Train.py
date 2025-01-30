import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

from GTSRB_CNN.data_pre_proc import load_data

tf.compat.v1.enable_eager_execution()
from tensorflow.compat.v1 import ConfigProto, InteractiveSession
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.activations import relu
from tensorflow.keras.initializers import he_normal, zeros, RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as keras_be

# Fix for internal CUDA error
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def get_model(img_size=(30, 30)):
    """
    Construct and return model
    """
    input_layer = Input(shape=(img_size[0], img_size[1], 3,), dtype='float32')
    cv1 = Conv2D(filters=32, kernel_size=5, activation=relu,
                 kernel_initializer=he_normal(), bias_initializer=zeros())(input_layer)
    cv2 = Conv2D(filters=64, kernel_size=3, activation=relu,
                 kernel_initializer=he_normal(), bias_initializer=zeros())(cv1)
    mp1 = MaxPool2D(pool_size=(2, 2))(cv2)
    do1 = Dropout(0.25)(mp1)
    cv3 = Conv2D(filters=64, kernel_size=3, activation=relu,
                 kernel_initializer=he_normal(), bias_initializer=zeros())(do1)
    mp2 = MaxPool2D(pool_size=(2, 2))(cv3)
    do2 = Dropout(0.25)(mp2)

    flat = Flatten()(do2)
    fc1 = Dense(units=256, activation=relu, kernel_initializer=he_normal(),
                bias_initializer=zeros(), kernel_regularizer=l2(1e-3))(flat)
    do3 = Dropout(0.5)(fc1)

    # Outputs
    cf = Dense(units=43, activation=None, name="classification",
               kernel_regularizer=l2(1e-4))(do3)
    reg = Dense(units=4, activation='linear', name="regression",
                kernel_initializer=RandomNormal(), kernel_regularizer=l2(0.1))(do3)

    return Model(inputs=input_layer, outputs=[cf, reg])


def plot_training(history, filename='training.png', save_dir='data'):
    # Create four side-by-side subplots
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    # 1. Classification Loss
    axs[0].plot(history.history["classification_loss"], label="train")
    axs[0].plot(history.history["val_classification_loss"], label="val")
    axs[0].set_title("Classification Loss")
    axs[0].set_xlabel("epochs")
    axs[0].set_ylabel("loss")
    axs[0].legend()

    # 2. Regression Loss
    axs[1].plot(history.history["regression_loss"], label="train")
    axs[1].plot(history.history["val_regression_loss"], label="val")
    axs[1].set_title("Regression Loss")
    axs[1].set_xlabel("epochs")
    axs[1].set_ylabel("loss")
    axs[1].legend()

    # 3. Classification Accuracy
    # Note: Depending on TensorFlow/Keras version, the metric might be
    # 'classification_accuracy' or 'classification_acc'.
    axs[2].plot(history.history["classification_acc"], label="train")
    axs[2].plot(history.history["val_classification_acc"], label="val")
    axs[2].set_title("Classification Accuracy")
    axs[2].set_xlabel("epochs")
    axs[2].set_ylabel("accuracy")
    axs[2].legend()

    # 4. Regression R^2
    axs[3].plot(history.history["regression_r2_keras"], label="train")
    axs[3].plot(history.history["val_regression_r2_keras"], label="val")
    axs[3].set_title("Regression R²")
    axs[3].set_xlabel("epochs")
    axs[3].set_ylabel("R²")
    axs[3].legend()

    plt.tight_layout()
    out_path = os.path.join(save_dir, filename)
    plt.savefig(out_path)


def r2_keras(y_true, y_pred):
    """
    Coefficient of determination for regression model
    """
    ss_res = keras_be.sum(keras_be.square(y_true - y_pred))
    ss_tot = keras_be.sum(keras_be.square(y_true - keras_be.mean(y_true)))
    return 1 - ss_res / (ss_tot + keras_be.epsilon())


def train_main(train_path='../data/train.npy', test_path='../data/test.npy', image_size=(100, 100),
               weights_path='result.weights.h5', save_dir='../data'):
    # Load data
    trainX, trainRoiY, trainLabelY = load_data(train_path)
    testX, testRoiY, testLabelY = load_data(test_path)

    model = get_model(image_size)

    loss = SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        optimizer="adam",
        loss={"classification": loss, "regression": "mse"},
        metrics={"classification": "acc", "regression": r2_keras},
        loss_weights={"classification": 5, "regression": 1}
    )

    # Train for more epochs to visualize a meaningful curve
    history = model.fit(
        x=trainX,
        y={"regression": trainRoiY, "classification": trainLabelY},
        validation_data=(testX, {"regression": testRoiY, "classification": testLabelY}),
        epochs=1,
        batch_size=32,
        verbose=1
    )

    model.save_weights(weights_path)

    plot_training(history, filename='training.png', save_dir=save_dir)


if __name__ == '__main__':
    train_main()
