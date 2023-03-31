import numpy as np
import click
import warnings
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
import seaborn as sns
import pandas as pd

from keras.models import Sequential, load_model
from keras.layers import (
    Dense,
    Flatten,
    Dropout,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
)
from keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
)
from sklearn.metrics import confusion_matrix, classification_report

warnings.filterwarnings(
    "ignore", message="Found untraced functions such as _jit_compiled_convolution_op"
)

from keras.callbacks import Callback
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
)
import numpy as np
import csv


def import_train_test(input, output, proportion):
    with open(input, "rb") as f:
        X = np.load(f)

    with open(output, "rb") as f:
        y = np.load(f)

    train_proportion = int(proportion * len(X))
    X_train = X[:train_proportion]
    X_test = X[train_proportion:]
    y_train = y[:train_proportion]
    y_test = y[train_proportion:]
    return X_train, X_test, y_test, y_train


def create_model(size_image_input, lr):
    classifier = Sequential()

    classifier.add(
        Conv2D(
            64,
            (3, 3),
            input_shape=(size_image_input, size_image_input, 3),
            activation="relu",
        )
    )
    classifier.add(Conv2D(64, (3, 3), activation="relu"))
    classifier.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    classifier.add(Conv2D(128, (3, 3), activation="relu"))
    classifier.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    classifier.add(Conv2D(256, (3, 3), activation="relu"))
    classifier.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    classifier.add(Conv2D(512, (3, 3), activation="relu"))
    classifier.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    classifier.add(Conv2D(512, (3, 3), activation="relu"))
    classifier.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    classifier.add(Flatten())
    classifier.add(Dense(units=256, activation="relu"))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(units=128, activation="relu"))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(units=1, activation="sigmoid"))

    classifier.compile(
        optimizer=Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", "AUC"],
    )
    return classifier


@click.command()
@click.argument("input_filepath", type=str)
@click.argument("output_filepath", type=str)
@click.argument("size_image_input", type=int)
@click.argument("model_name", type=str)
def main(input_filepath, output_filepath, size_image_input, model_name):
    # Train
    X_train, X_test, y_test, y_train = import_train_test(
        "data/processed/" + input_filepath,
        "data/processed/" + output_filepath,
        PROPORTION_TRAIN_TEST,
    )
    log_dir = (
        "./models/train_"
        + model_name
        + "/"
        + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    )
    # checkpoint = ModelCheckpoint('models/callback_checkpoint/model', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    model = create_model(size_image_input, LEARNING_RATE)
    h = model.fit(
        X_train,
        y_train[:, 0],
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        validation_split=VALIDATION_SPLIT,
        callbacks=[tensorboard_callback],
        verbose=1,
    )
    model.save("./models/" + model_name)

    # Validation
    y_pred = []
    for i in range(len(X_test)):
        is_front = model.predict(X_test[i : i + 1])
        if is_front < 0.5:
            y_pred.append(0)
        else:
            y_pred.append(1)

    results = model.evaluate(X_test, y_test[:, 0], verbose=0)

    # Print Results
    print(
        "Loss: {:.3f}, Accuracy: {:.3f}, AUC-ROC: {:.3f}".format(
            results[0], results[1], results[2]
        )
    )

    # Print Confusion matrix
    print("Confusion matrix:")
    print(confusion_matrix(y_test[:, 0], y_pred))

    # Save confusion matrix as a png
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test[:, 0], y_pred), annot=True, fmt="d")
    plt.title("Matrice de confusion")
    plt.xlabel("Prédits")
    plt.ylabel("Vraies valeurs")
    plt.savefig("./models/train_" + model_name + "/confusion_matrix.png")

    # Print Classification report
    print("Classification report:")
    print(classification_report(y_test[:, 0], y_pred, zero_division=0))
    # Conversion du rapport de classification en DataFrame Pandas
    df_report = pd.DataFrame(
        classification_report(y_test[:, 0], y_pred, output_dict=True)
    ).transpose()

    # Save Classification report in a CSV file
    df_report.to_csv(
        "./models/train_" + model_name + "/classification_report.csv", index=True
    )


NUM_EPOCHS = 2
BATCH_SIZE = 10
VALIDATION_SPLIT = 0.1
PROPORTION_TRAIN_TEST = 0.8
LEARNING_RATE = 0.001

if __name__ == "__main__":
    main()
