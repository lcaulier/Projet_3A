import torch
import clip
import numpy as np
import click
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image
from keras.models import load_model
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from sklearn.metrics import confusion_matrix, classification_report


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("RN50x16", device=device)


def import_train_test(input, output):
    with open(input, "rb") as f:
        X = np.load(f)

    with open(output, "rb") as f:
        y = np.load(f)
    return X, y


def ingredients_or_nutrition(img_to_process):
    img = (
        preprocess(Image.fromarray((img_to_process).astype("uint8")))
        .unsqueeze(0)
        .to(device)
    )
    text = clip.tokenize(["ingredient", "nutritional value"]).to(device)

    with torch.no_grad():
        logits_per_image, logits_per_text = model(img, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    return np.argmax(probs) + 1


@click.command()
@click.argument("input", type=str)
@click.argument("output", type=str)
@click.argument("model_name", type=str)
def main(model_name, input, output):
    classifier = load_model("models/models/" + model_name)
    X, y = import_train_test("data/processed/" + input, "data/processed/" + output)
    y_true = np.argmax(y, axis=1)
    y_pred = []
    for i in range(len(X)):
        img_to_process = X[i]
        is_front = classifier.predict(X[i : i + 1])
        if is_front < 0.5:
            class_predicted = np.eye(3)[ingredients_or_nutrition(img_to_process)]
        else:
            class_predicted = np.eye(3)[0]
        y_pred.append(np.argmax(class_predicted))

    # Print confusion matrix as a png
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d")
    plt.title("Matrice de confusion")
    plt.xlabel("Prédits")
    plt.ylabel("Vraies valeurs")
    plt.show()
    plt.savefig("./src/visualization/confusion_matrix.png")

    # Print Classification report
    print("Classification report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    # Save Classification report in a CSV file
    df_report = pd.DataFrame(
        classification_report(y_true, y_pred, output_dict=True)
    ).transpose()
    df_report.to_csv("./src/visualization/classification_report.csv", index=True)
    return


main()
