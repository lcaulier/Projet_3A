# -*- coding: utf-8 -*-
# Need to launch make_datasset.py from path ../Projet_3A with command :
# python src\data\make_dataset.py data\external\en.openfoodfacts.org.products.cs 224

import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import os
import numpy as np
import requests
from random import randint
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array


@click.command()
@click.argument("csv_filepath", type=click.Path(exists=True))
@click.argument("size_image_output", type=int)
def main(csv_filepath, size_image_output):
    """Import data from en.openfoodfacts.org.products.csv in (../raw) then
    runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("Importing data from CSV")

    path_foldername_front = "data/raw/images_dataset_RVB/images_front"
    path_foldername_nutrition = "data/raw/images_dataset_RVB/images_nutrition"
    path_foldername_ingredients = "data/raw/images_dataset_RVB/images_ingredients"

    i = 0  # counter to parse the file
    step = 100  # step to the next rows
    x = 0  # number of front images
    y = 0  # number of ingredients images
    z = 0  # number of nutritional images
    limit = 1  # If we want 2000 images per categories

    while x < limit or y < limit or z < limit:
        random = randint(i, i + 100)
        data = pd.read_csv(csv_filepath, sep="\t", skiprows=random, nrows=1)
        prod_id = data.iloc[0, 0]
        if not pd.isna(data.iloc[0, 79]) and x < limit:
            filepath = os.path.join(
                path_foldername_front, "front_" + str(prod_id) + ".jpg"
            )
            rep = requests.get(data.iloc[0, 79])
            with open(filepath, "wb") as f:
                f.write(rep.content)
            x += 1
        if not pd.isna(data.iloc[0, 81]) and y < limit:
            filepath = os.path.join(
                path_foldername_ingredients, "ingredients_" + str(prod_id) + ".jpg"
            )
            rep = requests.get(data.iloc[0, 81])
            with open(filepath, "wb") as f:
                f.write(rep.content)
            y += 1
        if not pd.isna(data.iloc[0, 83]) and z < limit:
            filepath = os.path.join(
                path_foldername_nutrition, "nutrition_" + str(prod_id) + ".jpg"
            )
            rep = requests.get(data.iloc[0, 83])
            with open(filepath, "wb") as f:
                f.write(rep.content)
            z += 1

        i += step

    logger.info("making final data set from raw data")

    # Images to array
    front_arrays = []
    foldername_front = os.listdir(path_foldername_front)
    for filename in foldername_front:
        front_arrays.append(
            img_to_array(
                load_img(
                    path_foldername_front + "/" + filename,
                    target_size=(size_image_output, size_image_output),
                )
            )
        )

    nutrition_arrays = []
    foldername_nutrition = os.listdir(path_foldername_nutrition)
    for filename in foldername_nutrition:
        nutrition_arrays.append(
            img_to_array(
                load_img(
                    path_foldername_nutrition + "/" + filename,
                    target_size=(size_image_output, size_image_output),
                )
            )
        )

    ingredients_arrays = []
    foldername_ingredients = os.listdir(path_foldername_ingredients)
    for filename in foldername_ingredients:
        ingredients_arrays.append(
            img_to_array(
                load_img(
                    path_foldername_ingredients + "/" + filename,
                    target_size=(size_image_output, size_image_output),
                )
            )
        )

    # Fusion of the 3 categories
    X = np.concatenate((front_arrays, ingredients_arrays, nutrition_arrays))
    y = np.concatenate(
        (
            np.array([[1, 0, 0]] * len(front_arrays)),
            np.array([[0, 1, 0]] * len(ingredients_arrays)),
            np.array([[0, 0, 1]] * len(front_arrays)),
        )
    )
    p = np.random.permutation(len(y))

    X = X[p]
    y = y[p]

    # Saving the data to save RAM
    with open("data/processed/input_" + str(size_image_output) + ".npy", "wb") as f:
        np.save(f, X)

    with open("data/processed/output_" + str(size_image_output) + ".npy", "wb") as f:
        np.save(f, y)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
