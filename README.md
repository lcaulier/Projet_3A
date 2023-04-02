
Projet_3A
==============================

Classification (front, ingredients, nutritional values) of Open Food Facts Images to detect errors in the database.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>





## Installation

Install my-project’s required python packages with pip

```bash
  cd my-project
  pip install requirements.txt
```
You will also need to have software installed to run and execute a Jupyter Notebook.

## Create Dataset

The first step is to create a set of Open Food Facts images classified in 3 categories: Front, Ingredients, Nutritional Values.

To do this, we will have to download the CSV export in .gz from the Open Food Facts website and put it in data/external.

Then the command below allows you to classify the images in data/raw with a square format to be defined in the command.

```bash
  python src\data\make_dataset.py data\external\en.openfoodfacts.org.products.cs "Image sizes"
```

These preprocessed images are stored in data/preprocessed as input_image_sizes.npy and output_image_sizes.npy. These 2 files correspond to the X and Y to provide to our deep learning models.
This avoids RAM problems on local computers.



## CNN training

The command below allows to launch a training of the CNN allowing to classify if an image is front or not. The first argument following the file train.py corresponds to the **input file** at the input of the model, the second to the **output**, the third to the **size of the images** on which our model will train and the last to the **name of our training**.
In this example we had saved inputs and outputs in size 224 and we named this training test1.

```bash
python  src/models/train_model.py  input_array_224.npy  output_array_224.npy  224  test1
```
Inside the **_train.py_** file many training parameters can be modified such as the number of _epochs_, the _batchsize_, the _learning rate_, the _validation split_.

The results of the training will be stored in the **_/models_** folder. A first sub-folder named **_train+model_name_** contains the metrics that can be viewed in tensorboard with the command below (to be adapted according to the name given to the training). In the same sub-folder, there is the confusion matrix, the classification ratio and the hyperparameters used for training.

```bash
tensorboard --logdir models/test1
```

A second sub-folder named **_model_name_** corresponds to the saved Keras model which can then be reused later.

## Visualization

The command below launches the program visualization.py which returns a classification report and a confusion matrix on the 3 classes (front, ingredients, nutritional values).

```bash
python  src/models/visualize.py  input_array_224.npy  output_array_224.npy  model_name
```

## Notebook

In the notebook folder, we have collected all the jupyter notebooks that were useful to us in creating these models as well as some of the explorations we made. In the notebooks there is a brief description of its use.
