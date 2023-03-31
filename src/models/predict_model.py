import torch
import clip
import numpy as np
import click
from PIL import Image
from keras.models import load_model
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("RN50x16", device=device)


def ingredients_or_nutrition(img_to_process):
    img = (
        preprocess(Image.fromarray((img_to_process).astype("uint8")))
        .unsqueeze(0)
        .to(device)
    )
    text = clip.tokenize(["ingredient", "nutritional value"]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(img)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(img, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    return np.argmax(probs) + 1


@click.command()
@click.argument("model_name", type=str)
@click.argument("image", type=str)
def main(image, model_name):
    classifier = load_model("models/saved_models/" + model_name)
    img_to_process = load_img(image, target_size=(size_image_input, size_image_input))
    img_expand = np.expand_dims(img_to_process, axis=0)
    is_front = classifier.predict(img_expand)
    if is_front < 0.5:
        class_predicted = np.eye(3)[ingredients_or_nutrition(img_to_process)]
    else:
        class_predicted = np.eye(3)[0]
    print(class_predicted)
    return class_predicted


size_image_input = 224
main()
